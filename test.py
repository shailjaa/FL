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
import zipfile
from flask import Flask,request,Response
app = Flask(__name__)

s3_client = boto3.client('s3', 
                      aws_access_key_id='AKIATOXCR6TP47PYZPW7', 
                      aws_secret_access_key='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')
base_directory = "./dataset/process_data/"

def chunking(request_data):
    my_bucket = request_data['bucket_name']
    total_jobs = ast.literal_eval(request_data['total_jobs'])
    s3_client.download_file(my_bucket, request_data['dataset'], request_data['dataset'].split("/")[-1])
    with zipfile.ZipFile(request_data['dataset'].split("/")[-1], 'r') as zip_ref:
        zip_ref.extractall("./dataset/")

    print("./dataset/" + request_data['dataset'].split(".")[0] + "/")
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.SBU(base_directory + "actual_data/", download = True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=1000, shuffle=True)

    examples = enumerate(train_loader)
    print(len(train_loader))
    
    data_batch_list = []
    
    os.makedirs(base_directory + "data/")
#     os.makedirs(base_directory + "target/")
    os.makedirs(base_directory + "Zip_data/")

    for i in range(len(train_loader)):
        batch_idx, data = next(examples)
        data_batch_list.append(batch_idx)
        torch.save(data, base_directory + "data/" + str(batch_idx)+".pt")
#         torch.save(target, base_directory + "target/"+str(batch_idx)+".pt")

    count = 0
    if total_jobs < 600:
        data_length=math.ceil(len(train_loader)/total_jobs)
        for i in range(int(total_jobs)):
            os.makedirs(base_directory + str(i) + "/data/")
#             os.makedirs(base_directory + str(i) + "/target/")
            for j in range(int(data_length)):
                print("COUNT", count)
                if count == len(train_loader):
                    pass
                else : 
                    random_pt = random.choice(data_batch_list)
                    data_batch_list.remove(random_pt)
                    shutil.copy2(base_directory + "data/" + str(random_pt)+".pt", base_directory + str(i)+"/data/" + str(random_pt) + ".pt")
#                     shutil.copy2(base_directory + "target/" + str(random_pt)+".pt", base_directory + str(i)+"/target/" + str(random_pt) + ".pt")
                    count = count + 1

            make_archive(base_directory + str(i), base_directory + "Zip_data/" + str(i)+ ".zip")
            s3_client.upload_file( base_directory + "Zip_data/" + str(i)+ ".zip", request_data['bucket_name'], "data/" + str(i) + ".zip")
    # except Exception as e:
    #     print("Error : ", e)
    #     traceback.print_exc

def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)

@app.route('/chunk_data', methods=['POST'])
def index():    
    try:
        if request.method == "POST":
            
            print(request.form, request)
            request_data = request.form.to_dict()
            chunking(request_data)

            if(os.path.exists('./dataset/')):
                shutil.rmtree('./dataset/')
            return flask.jsonify({'code':200, 'Message' : 'Data Chunking Done Successfully'})

        else:
            if(os.path.exists('./dataset/')):
                shutil.rmtree('./dataset/')
            return flask.jsonify({'code':400, 'Message' : 'Some Input Value is not correct'})

    except Exception as e:
        print("Exception in Average Server", e)
        traceback.print_exc()
        if(os.path.exists('./dataset/')):
            shutil.rmtree('./dataset/')
        return flask.jsonify({'code':400, 'Message':'Data Not Uploaded successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)
