import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable

from check_installed import check_installed_libs
from upload_train_data import upload_train_data
from train import train
from train import upload_model_and_remove_dir

dag = DAG(
    dag_id='prepare_model_dag',
    schedule_interval=timedelta(hours=3),
    start_date=datetime(2024, 6, 14, 11, 20, 0),
    tags=['final_project'],
)

check_install = PythonOperator(task_id='check_installed',
                                python_callable=check_installed_libs,
                                dag=dag)

check_new_objs = PythonOperator(task_id='upload_train_data',
                                python_callable=upload_train_data,
                                dag=dag)

train = PythonOperator(task_id='train',
                                python_callable=train,
                                dag=dag)
upload_model = PythonOperator(task_id='upload_model',
                                python_callable=upload_model_and_remove_dir,
                                dag=dag) 

check_install >> check_new_objs >> train >> upload_model