import os
import json
import math
import torch
import shutil
import random
import argparse
import traceback
import torchvision
import importlib.util

import boto3

s3 = boto3.resource('s3')
BUCKET = "federated-learning-test"

s3.Bucket(BUCKET).upload_file("your/local/file", "dump/file")


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("actual_data/" , train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=7500, shuffle=True)

examples = enumerate(train_loader)

for i in range(len(train_loader)):
    batch_idx, (data, target) = next(examples)
    torch.save(data, "data/" + str(batch_idx)+".pt")
    torch.save(target, "target/"+str(batch_idx)+".pt")
    s3.Bucket(BUCKET).upload_file("data/" + str(batch_idx) + ".pt", "data/" + str(batch_idx) + ".pt")
    s3.Bucket(BUCKET).upload_file("target/" + str(batch_idx) + ".pt", "target/" + str(batch_idx) + ".pt")   
