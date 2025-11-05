# Image-Finetuners

Our objective is to finetune the image classification foundation model.

---

## Base Foundation Model

The base foundation model is **DINOv2** trained by Meta.  
[🔗 Hugging Face model link](https://huggingface.co/facebook/dinov2-small)



## Dataset

We use the **Facial Emotion Expressions** dataset from Kaggle.  
[👉 View on Kaggle](https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions)

This dataset contains facial images categorized by emotion labels such as *angry*, *happy*, *sad*, *neutral*, and more.  
It is commonly used for emotion recognition and facial expression classification tasks.

To download the dataset, you can use the **kagglehub** Python package.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small")

import kagglehub

path = kagglehub.dataset_download("samaneheslamifar/facial-emotion-expressions")
print("Path to dataset files:", path)

