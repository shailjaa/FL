import time, os, sys
import sagemaker, boto3
import numpy as np
import pandas as pd
import os
import json
import math
import shutil
import random
import argparse
import traceback
import importlib.util
import torch
import torchvision
import boto3

# sess = boto3.Session()
# sm   = sess.client('sagemaker')
# role = sagemaker.get_execution_role()
# sagemaker_session = sagemaker.Session(boto_session=sess)

# datasets = sagemaker_session.upload_data(path='data/', key_prefix='data_1/')

BUCKET_NAME = 'federated-learning-testing' # replace with your bucket name
KEY = 'data_1/cat/cat.0.jpg' # replace with your object key

s3 = boto3.resource('s3')
s3_client = boto3.client('s3', 
                      aws_access_key_id='AKIATOXCR6TP47PYZPW7', 
                      aws_secret_access_key='KBSpnYigeYG4UAafw2aMGf3MnCwetA0IVFvLyThB')

my_bucket = s3.Bucket(BUCKET_NAME)

if not os.path.exists("data_1/dog"):
    os.makedirs("data_1/dog")
if not os.path.exists("data_1/cat"):
    os.makedirs("data_1/cat")

bucket_prefix_cat="data_1/cat/" 
bucket_prefix_dog="data_1/dog/" 

for objects in my_bucket.objects.filter(Prefix=bucket_prefix_cat):
    print(objects.key)
    my_bucket.download_file(objects.key, "data_1/cat/" + objects.key.split("/")[-1])

for objects in my_bucket.objects.filter(Prefix=bucket_prefix_dog):
    print(objects.key)
    my_bucket.download_file(objects.key, "data_1/dog/" + objects.key.split("/")[-1])

data = torchvision.datasets.ImageFolder(root = "data_1/",  transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]))

train_loader = torch.utils.data.DataLoader(data,batch_size=2, shuffle=True)

examples = enumerate(train_loader)

j = 0
for i in range(len(train_loader)):
    if(i % 10 == 0):
        j = j+1
        os.makedirs("./data/" + str(j) + "/data/")
        os.makedirs("./data/" + str(j) + "/target/")
        print(j)
    batch_idx, (data, target) = next(examples)
    torch.save(data, "./data/" + str(j) + "/data/" + str(batch_idx)+".pt")
    torch.save(target,"./data/" + str(j) + "/target/" + str(batch_idx)+".pt") 
    s3_client.upload_file("./data/" + str(j) + "/data/" + str(batch_idx) + ".pt", BUCKET_NAME, "data/" + str(j) + "/data/" + str(batch_idx) + ".pt")
    s3_client.upload_file("./data/" + str(j) + "/target/" +  str(batch_idx) + ".pt", BUCKET_NAME, "data/" + str(j) + "/target/" + str(batch_idx) + ".pt")