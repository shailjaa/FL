import importlib
import os
import math
import shutil
import torch
import boto3
import flask
import traceback
import ast
from flask import Flask, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ADD S3 Client Access and Secret Key for Data Uploading and Downloading
s3_client = boto3.client('s3',
                         aws_access_key_id='AKIATOXCR6TP47PYZPW7',
                         aws_secret_access_key='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')

# Base Directory to process data
base_directory = "./dataset/process_data/"

# Create list with file names and classnames
def default_flist_reader(filelist, classlist):
    file_class_list = []
    for item in filelist:
        image_path = item
        print(item)
        image_label = item.split("/")[-2]
        file_class_list.append((image_path, int(classlist.index(image_label))))
    print("flisdt : ", file_class_list)
    return file_class_list

# create zip file to upload single file containing data for single worker in s3
def create_chunk_zip(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)

# Create zip of all pt files for one job and upload it in s3
def create_zip_and_upload(base_directory, request_data, i, examples):
    os.makedirs(base_directory + str(i) + "/data/")
    os.makedirs(base_directory + str(i) + "/target/")
    for j in range(5):
        batch_idx, (data, target) = next(examples)
        torch.save(data, base_directory + str(i) + "/data/" + str(batch_idx) + ".pt")
        torch.save(target, base_directory + str(i) + "/target/" + str(batch_idx) + ".pt")
    create_chunk_zip(base_directory + str(i), base_directory + "Zip_data/" + str(i) + ".zip")
    s3_client.upload_file(base_directory + "Zip_data/" + str(i) + ".zip", "federated-learning-testing",
                            request_data["project_upload_path"] + str(i) + ".zip")
    
# Chunk Loaders
def chunking(request_data):

    s3 = boto3.resource('s3')

    # Get required general fields from request
    bucket = s3.Bucket( request_data['bucket_name'])
    total_jobs = ast.literal_eval(request_data['total_jobs'])
    dataset_type = request_data['dataset_type']
    data_loader_file = request.files["data_loader"]
    data_loader_filename = secure_filename(data_loader_file.filename)  # save surface csv locally to use

    if data_loader_filename == '':
        return jsonify({"Error": "No data loader file in data loader"})  # if no surface csv found throw error

    data_loader_filepath = os.path.join(data_loader_filename)  # join surface csv path
    data_loader_file.save(data_loader_filepath)

    # stores zip file from which data will be uploaded to s3
    os.makedirs(base_directory + "/Zip_data/")

    # For any Pytorch library dataset, this type allows to chunk
    if dataset_type == "pytorch_general":
        dataloader_spec = importlib.util.spec_from_file_location("returnTrainLoader", data_loader_filepath)
        dataloader_file_object = importlib.util.module_from_spec(dataloader_spec)
        dataloader_spec.loader.exec_module(dataloader_file_object)

        # get train loader from data loader file path added
        train_loader = dataloader_file_object.returnTrainLoader(25, "./Classic_Dataset/")
        examples = enumerate(train_loader)
        data_length = math.ceil(len(train_loader)/total_jobs)

        if total_jobs < 600:
            for i in range(data_length):
                create_zip_and_upload(base_directory, request_data, i, examples)

    # This type allows us to chunk data for classification
    if dataset_type == "pytorch_classification":
        files = [obj.key for obj in bucket.objects.filter(Prefix=request_data["data_path"])]
        datalist = default_flist_reader(files, ast.literal_eval(request_data["class_list"]))

        if total_jobs < len(files):
            data_length = math.ceil(len(files) / total_jobs)

        # chunk loader as many as jobs and store batches in zip
        for i in range(total_jobs):
            dataloader_spec = importlib.util.spec_from_file_location("returnTrainLoader", data_loader_filepath)
            dataloader_file_object = importlib.util.module_from_spec(dataloader_spec)
            dataloader_spec.loader.exec_module(dataloader_file_object)

            train_loader = dataloader_file_object.returnTrainLoader(datalist[i * data_length:(i + 1) * data_length], bucket)

            examples = enumerate(train_loader)
            create_zip_and_upload(base_directory, request_data, i, examples)

# -------------------- FLASK APP -------------------- #
# FLASK APP : chunk_data/ api called for chunking data

#  -- DEMO API FORM DATA  FOR CLASSIFICATION DATASET --
# dataset_type:pytorch_classification
# bucket_name:federated-learning-testing
# total_jobs:5
# data_loader:data_loader_classification.py
# data_path:data_1/data_1/
# class_list:["cat", "dog"]
# project_upload_path : user_id/project_id/data/

#  -- DEMO API FORM DATA  FOR PYTORCH DATASET --
# dataset_type:pytorch_general
# bucket_name:federated-learning-testing
# total_jobs:5
# data_loader:data_loader.py
# project_upload_path : user_id/project_id/data/

@app.route('/chunk_data', methods=['POST'])
def index():
    try:
        if request.method == "POST":

            print(request.form, request)
            request_data = request.form.to_dict()
            chunking(request_data)

            if (os.path.exists('./dataset/')):
                shutil.rmtree('./dataset/')
            return flask.jsonify({'code': 200, 'Message': 'Data Chunking Done Successfully'})

        else:
            if (os.path.exists('./dataset/')):
                shutil.rmtree('./dataset/')
            return flask.jsonify({'code': 400, 'Message': 'Some Input Value is not correct'})

    except Exception as e:
        print("Exception in Average Server", e)
        traceback.print_exc()
        if (os.path.exists('./dataset/')):
            shutil.rmtree('./dataset/')
        return flask.jsonify({'code': 400, 'Message': 'Data Not Uploaded successfully'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)