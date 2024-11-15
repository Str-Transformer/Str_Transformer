import argparse
import importlib
import math
import os
import socket
import sys
import torch.optim as optim

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

# from loss.pointnetvlad_loss import MDRLoss

import config as cfg
import evaluate
import loss.loss as PNV_loss

import models.Str_Transformer as St_Fomer


import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
import evaluate as ev
import torch.nn.functional as F
import time

import faiss


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

dataStructure = "KDTree"

cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of positive examples per query')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of negative examples per query')
parser.add_argument('--max_epoch', type=int, default=20,
                    help='Epoch to run [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=1, #2
                    help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.000005,
                    help='Initial learning rate [default: 0.000005]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5,
                    help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2,
                    help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--loss_function', default='quadruplet_loss', choices=[
                    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_false',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_true', default=False,
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='/home/ajou/lidarur/generating_queries/',
                    help='PointNetVlad Dataset Folder')

FLAGS = parser.parse_args()
cfg.BATCH_NUM_QUERIES = FLAGS.batch_num_queries
#cfg.EVAL_BATCH_SIZE = 1 #12
cfg.NUM_POINTS = 4096
cfg.TRAIN_POSITIVES_PER_QUERY = FLAGS.positives_per_query
cfg.TRAIN_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2
# cfg.FEATURE_OUTPUT_DIM = 256

cfg.LOSS_FUNCTION = FLAGS.loss_function
cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch

cfg.TRAIN_FILE = '/home/ajou/lidarur/generating_queries/training_queries_baseline_v1.pickle'
# '/home/mlmlab08/study/generating_queries/training_queries_baseline.pickle'
cfg.TEST_FILE = '/home/ajou/lidarur/generating_queries/test_queries_baseline_v1.pickle'
# '/home/mlmlab08/study/generating_queries/test_queries_baseline.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

cfg.RESULTS_FOLDER = FLAGS.results_dir

cfg.DATASET_FOLDER = FLAGS.dataset_folder

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
# breakpoint()
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
useSavedQuery = True


cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
        (cfg.BN_DECAY_DECAY_RATE **
         (batch * cfg.BATCH_NUM_QUERIES // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch


def get_learning_rate(epoch):
    learning_rate = cfg.BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    global best_ave_one_percent_recall
    best_ave_one_percent_recall = 0
    bn_decay = get_bn_decay(0)
    #tf.summary.scalar('bn_decay', bn_decay)

    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if cfg.LOSS_FUNCTION == 'triplet_loss':
        loss_function = PNV_loss.distance_quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
        
    learning_rate = get_learning_rate(0)

    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))
    #test_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'test'))

                          
    model = St_Fomer.Str_Transformer(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)

    model = model.to(device)
  

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if FLAGS.resume:
        resume_filename = cfg.LOG_DIR + cfg.MODEL_FILENAME
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)

        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    model = nn.DataParallel(model)
    


    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    
    queries_list, edge_queries_list = get_queries(TRAINING_QUERIES)

    for epoch in range(starting_epoch, cfg.MAX_EPOCH):
        print(epoch)
        print()
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(model, optimizer, train_writer, loss_function, epoch, queries_list, edge_queries_list)

        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + cfg.OUTPUT_FILE_NAME + str(epoch) + '.txt'
        eval_recall = evaluate.evaluate_model(model)
        log_string('EVAL RECALL: %s' % str(eval_recall))

        train_writer.add_scalar("Val Recall", eval_recall, epoch)

best_ave_one_percent_recall = float('-inf')

def train_one_epoch(model, optimizer, train_writer, loss_function, epoch, queries_list, edge_queries_list):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS
    global best_ave_one_percent_recall
    print(TRAINING_LATENT_VECTORS)

    is_training = True
    sampled_neg = 4000


    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    for i in range(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES):
        
        def get_random_number(mean=4, std_dev=1, min_val=2, max_val=18):
            number = np.random.normal(mean, std_dev)
            number = np.clip(number, min_val, max_val)
            return int(number)
        
        positives_per_query = get_random_number()
        negatives_per_query = 20-positives_per_query
        num_to_take = int(math.floor(negatives_per_query / 2))

        # for i in range (5):
        batch_keys = train_file_idxs[i * cfg.BATCH_NUM_QUERIES:(i+1)*cfg.BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        
        batch_losses = []
        angle_losses = []
        for j in range(cfg.BATCH_NUM_QUERIES):
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < positives_per_query):
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], positives_per_query, negatives_per_query,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif (len(HARD_NEGATIVES.keys()) == 0):
                st = time.time()
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                
                print(hard_negs)

                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], positives_per_query, negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))

                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], positives_per_query, negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            #     # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            #     # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

                print("3")
                for k in range(len(q_tuples)):
                    print(q_tuples[k][0].shape)
                    print(q_tuples[k][1].shape)
                    print(q_tuples[k][2].shape)
                    print(q_tuples[k][3].shape)

            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
                no_other_neg = True
                break

        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)
        
        # print(queries.shape, other_neg.shape, positives.shape, negatives.shape)
        log_string('----' + str(i) + '-----')
        if (len(queries.shape) != 4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg,positives_per_query ,negatives_per_query)

        loss_distance = PNV_loss.distance_quadruplet_loss(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, 
                                                    use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)

        angle_rank = PNV_loss.MWUAngularLoss(device='cuda')
        angle_rank_loss = angle_rank(output_queries, output_positives, output_negatives, output_other_neg)
 
        total_loss = loss_distance + angle_rank_loss

        total_loss.backward()
    
        optimizer.step()
        
        batch_losses.append(total_loss.item())
        angle_losses.append(angle_rank_loss.item())
        
        
        if i>0 and i%(len(train_file_idxs)//360)==0:
            print(f"==== For {len(train_file_idxs)//360}'s loss mean ====")
            log_string('batch loss: %f' % (np.mean(batch_losses)))
            batch_losses, angle_losses = [], []
        else:
            log_string('batch loss: %f' % (np.mean(batch_losses)))

        train_writer.add_scalar("Loss", total_loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES
        
        

#best.model
        if (epoch >= 5 and i % (1400 // cfg.BATCH_NUM_QUERIES) == 29):
            if useSavedQuery:
                TRAINING_LATENT_VECTORS = get_latent_vectors(
                    model, TRAINING_QUERIES, positives_per_query, negatives_per_query)
            else:
                TRAINING_LATENT_VECTORS = get_latent_vectors_v2(
                    model, TRAINING_QUERIES, positives_per_query, negatives_per_query, queries_list, edge_queries_list)
            print("Updated cached feature vectors")
            

# 모델 성능 평가 및 최고 성능 모델 저장
        if (i % (5000 // cfg.BATCH_NUM_QUERIES) == 500 and i>0):  # 원하는 평가 주기
            print("=======================")
            print("Evaluated Starts")
            print("=======================")
            ave_one_percent_recall = ev.evaluate_model(model)  # 모델 평가 함수 호출
            if ave_one_percent_recall > best_ave_one_percent_recall:  # 성능이 개선된 경우에만 저장
                best_ave_one_percent_recall = ave_one_percent_recall
                if isinstance(model, nn.DataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model
                save_name = cfg.LOG_DIR + cfg.MODEL_FILENAME
                torch.save({
                    'epoch': epoch,
                    'iter': TOTAL_ITERATIONS,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, save_name)
                print(f"Model Saved As {save_name} with ave_one_percent_recall {ave_one_percent_recall}")
            else:
                print(f'ave_one_percent_recall {ave_one_percent_recall}')


def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output

def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])
    latent_vecs = np.array(latent_vecs)
    
    if dataStructure == "FAISS":
        index = faiss.IndexFlatL2(latent_vecs.shape[1])
        index.add(latent_vecs)
        distances, indices = index.search(np.array([query_vec]), num_to_take)
    elif dataStructure == "KDTree":
        nbrs = KDTree(latent_vecs)
        # et = time.time()
        distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs

def get_queries(dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    
    batch_num = cfg.BATCH_NUM_QUERIES * (22)
    
    queries_list = []; edge_queries_list = []
    
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)
        
        queries_list.append(queries)
        
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)
        
        edge_queries_list.append(queries)
        
    return queries_list, edge_queries_list

def get_latent_vectors_v2(model, dict_to_process,positives_per_query,negatives_per_query, queries_list, edge_queries_list):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + positives_per_query+ negatives_per_query+ 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        queries = queries_list[q_index]

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        queries = edge_queries_list[q_index]
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()

    return q_output

def get_latent_vectors(model, dict_to_process,positives_per_query,negatives_per_query):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))


    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + positives_per_query+ negatives_per_query+ 1)
    q_output = []
    model.eval()
    for q_index in range(len(train_file_idxs)//batch_num):
        st = time.time()
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        
        with torch.no_grad():
            out = model(feed_tensor)
            

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)
        # if (BATCH_NUM_QUERIES - 1 > 0):
        #    fake_queries = np.zeros((BATCH_NUM_QUERIES - 1, 1, NUM_POINTS, 3))
        #    q = np.vstack((queries, fake_queries))
        # else:
        #    q = queries

        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 3))
        #o1, o2, o3, o4 = run_model(model, q, fake_pos, fake_neg, fake_other_neg)
        
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)
            

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output
            
    model.train()
    # print(q_output.shape)
    return q_output

def run_model(model, queries, positives, negatives, other_neg,positives_per_query ,negatives_per_query,require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()

    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, cfg.NUM_POINTS, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    # print(feed_tensor.shape)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    
    o1, o2, o3, o4 = torch.split(
        output, [1, positives_per_query, negatives_per_query, 1], dim=1)

    return o1, o2, o3, o4

if __name__ == "__main__":
    train()








