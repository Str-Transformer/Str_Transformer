# Str_Transformer
Robust Place Recognition

Introduction
This paper proposes a step-wise cross-attention mechanism integrating 3D structural conv blocks and learnable queries for large-scale place recognition. The proposed architecture is designed to effectively capture structural features and local information from objects of various sizes and enhance performance by integrating global contextual information. Experimental results demonstrates that the proposed method outperforms existing state-of-the-art techniques on cross-source benchmark datasets. Unlike previous models, which may perform well on specific datasets but suffer significant performance degradation in other scenes, the proposed Structural transformer demonstrates the strongest generalization capabilities in cross-source datasets.

![image](https://github.com/user-attachments/assets/d2faeb65-f2b2-4281-bbab-2bb8d3aab984)


# We use the Oxford robotics benchmark datasets introduced in PointNetVLAD 

Oxford dataset
- NUS (in-house) Datasets
- university sector (U.S.)
- residential area (R.A.)
- business district (B.D.)


# Training and Evaluation
To train Str-Transformer model on the Baseline Dataset

- python train.py 

To evaluate pretrained Str-Transformer model on the Baseline Dataset

- python evaluate.py 

# Pre-trained Models
Pretrained models are available in logs
best_model.pth trained on the Baseline Dataset

# Acknowledgement
Our code refers to PointNetVLAD 
