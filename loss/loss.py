import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def distance_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # print('pos_vecs', pos_vecs.shape)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)

    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    # print(total_loss)
    return total_loss




class MWUAngularLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(MWUAngularLoss, self).__init__()
        self.device = device

    def forward(self, query, positive, negative, other_neg):

        q_vec = self.compute_vec(query, other_neg)
        pos_vec = self.compute_vec(positive, other_neg)
        neg_vec = self.compute_vec(negative, other_neg)
        
        pos_cosim = self.compute_cosine_similarity(q_vec, pos_vec)
        neg_cosim = self.compute_cosine_similarity(q_vec, neg_vec)
        
        loss = self.rank_based_loss(pos_cosim, neg_cosim)
    
        return loss

    def rank_based_loss(self, pos_angle, neg_angle):
        p_value_1, p_value_2 = self.mann_whitney_u(pos_angle, neg_angle)
        loss = (F.relu((1 - p_value_2)-0.05) + F.relu(p_value_1-0.05))
        loss = loss.mean()
        return loss

    def compute_cosine_similarity(self, vec_1, vec_2):
        vec_1_norm = F.normalize(vec_1, p=2, dim=2)
        vec_2_norm = F.normalize(vec_2, p=2, dim=2)
        angle = 2 - 2 * torch.bmm(vec_1_norm, vec_2_norm.transpose(1, 2))
        return angle

    def compute_vec(self, vec_1, vec_2):
        num_vecs = vec_1.shape[1]
        vec_2_copies = vec_2.repeat(1, num_vecs, 1)
        vectors = vec_1 - vec_2_copies
        return vectors
        
        
    def mann_whitney_u(self, group1, group2):
        group1 = group1.to(self.device)
        group2 = group2.to(self.device)

        combined = torch.cat([group1, group2], dim=-1)
        sorted_indices = torch.argsort(combined, dim=-1)
        ranks = torch.zeros_like(combined, dtype=torch.float32, device=self.device)

        # Calculate ranks without handling ties
        ranks.scatter_(-1, sorted_indices, torch.arange(1, combined.size(-1) + 1, device=self.device, dtype=torch.float32).unsqueeze(0).expand_as(sorted_indices))

        ranks1 = ranks[:, :, :group1.size(-1)]
        ranks2 = ranks[:, :, group1.size(-1):]

        # Calculate U values
        U1 = torch.sum(ranks1, dim=-1) - (group1.size(-1) * (group1.size(-1) + 1)) / 2
        U2 = torch.sum(ranks2, dim=-1) - (group2.size(-1) * (group2.size(-1) + 1)) / 2

        n1 = torch.tensor(group1.size(-1), dtype=torch.float32, device=self.device)
        n2 = torch.tensor(group2.size(-1), dtype=torch.float32, device=self.device)
        n = n1 + n2

        # Mean of U
        mu_U = n1 * n2 / 2

        # Standard deviation without ties
        sigma_U = torch.sqrt(n1 * n2 * (n + 1) / 12)

        # Z scores for one-tailed test (group1 < group2)
        z_1 = (U1 - mu_U) / sigma_U
        z_2 = (U2 - mu_U) / sigma_U

        # Calculate p-values using Z-scores
        p_value_1 = 0.5 * (1 + torch.erf(z_1 / torch.sqrt(torch.tensor(2.0, device=self.device))))
        p_value_2 = 0.5 * (1 + torch.erf(z_2 / torch.sqrt(torch.tensor(2.0, device=self.device))))

        return p_value_1, p_value_2


 

