import time, sys
import sagemaker
import numpy as np
import pandas as pd
import os
import json
import math
import shutil
import random
import torch
import torchvision
import boto3
import flask
import traceback
import ast
from flask import Flask,request,Response
app = Flask(__name__)

# BUCKET_NAME = 'federated-learning-testing' # replace with your bucket name

s3_client = boto3.client('s3', 
                      aws_access_key_id='AKIATOXCR6TP47PYZPW7', 
                      aws_secret_access_key='KBSpnYigeYG4UAafw2aMGf3MnCwetA0IVFvLyThB')
session = boto3.Session(aws_access_key_id='AKIATOXCR6TP47PYZPW7', 
                      aws_secret_access_key='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', 
                      )

def chunking(request_data):

    s3 = session.resource('s3')
    my_bucket = s3.Bucket(request_data['bucket_name'])

    for item in ast.literal_eval(request_data['classlist']):
        if not os.path.exists(item):
            os.makedirs(item)
        for objects in my_bucket.objects.filter(Prefix=item):
            print(objects.key)
            my_bucket.download_file(objects.key, item + objects.key.split("/")[-1])

    data = torchvision.datasets.ImageFolder(root = ast.literal_eval(request_data['classlist'])[0].split("/")[0],  transform=torchvision.transforms.Compose(
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
        s3_client.upload_file("./data/" + str(j) + "/data/" + str(batch_idx) + ".pt", request_data['bucket_name'], "data/" + str(j) + "/data/" + str(batch_idx) + ".pt")
        s3_client.upload_file("./data/" + str(j) + "/target/" +  str(batch_idx) + ".pt", request_data['bucket_name'], "data/" + str(j) + "/target/" + str(batch_idx) + ".pt")

@app.route('/chunk_data', methods=['POST'])
def index():
    try:
        if request.method == "POST":
            
            print(request.form, request)
            request_data = request.form.to_dict()
            chunking(request_data)

            return flask.jsonify({'code':200})
        else:
            return flask.jsonify({'code':400,'message':'Method must be POST'})
    except Exception as e:
        print("Exception in Average Server", e)
        traceback.print_exc()
        return flask.jsonify({'code':400,'message':'Path not found, Make sure processed model is available'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)

