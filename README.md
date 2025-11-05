# Image-Finetuners
Our objective is to finetune the image classification foundation model



## Load model directly
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")



# # Download dataset

Facial expression dataset: https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions

import kagglehub
path = kagglehub.dataset_download("samaneheslamifar/facial-emotion-expressions")
print("Path to dataset files:", path)
