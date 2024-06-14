import boto3
import pandas as pd
import numpy as np
import os

def upload_train_data():
    session = boto3.session.Session(aws_access_key_id = '***', 
                                    aws_secret_access_key='***')

    s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net')

    contents = s3.list_objects(Bucket='data-source-german-mlops')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='data-source-german-mlops')
    filelist_s3_source = []
    for page in pages:
        filelist_s3_source += [key['Key'] for key in page['Contents']]
    print('Count s3 content: ',len(filelist_s3_source))

    contents = s3.list_objects(Bucket='mlops-dev-german')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='mlops-dev-german')
    filelist_s3_dev = []
    for page in pages:
        filelist_s3_dev += [key['Key'] for key in page['Contents']]
    if 'train_list.csv' in filelist_s3_dev:
        s3.download_file('mlops-dev-german', 'train_list.csv', 'train_list.csv')
        file_list = pd.read_csv('train_list.csv')
        file_list = file_list['file_name'].tolist()
    else:
        file_list = []
    print('Count previous train files: ', len(set(file_list)))
    if len(set(filelist_s3_source) - set(file_list)) > 0:
        os.makedirs('data', exist_ok=True)
        for file_name in filelist_s3_source:
            s3.download_file('data-source-german-mlops', file_name, './data/'+file_name)

        file_list_new = [x for x in os.listdir('./data')]
        pd.DataFrame(file_list_new, columns = ['file_name']).to_csv('train_list.csv', index = None)
        s3.upload_file("train_list.csv",'mlops-dev-german',  "train_list.csv")