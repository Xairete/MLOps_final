# otus_mlops_final_project

Финальный проект по курсу MLOps: Сервис распознавания, еда или не еда на снимке. Реализовано на базе Yandex Cloud.    
Сервис реализован на Python с использованием Flask. Настроен мониторинг с использованием Prometeus и Grafana, которые, как и сам сервис, обернуты в Docker-контейнеры.  
Также реализованы DAG Airflow для автоматического переобучения при добавлении новых исходных данных.

## Содержимое каталога:
app - реализауия самого сервиса  
dags - DAG и вспомогательные скрипты  
grafana - дашборд Графаны  
prometeus - конфиг прометеуса (но сам он вызывается в скрипте приложения)    

## Несколько картинок
![image](https://github.com/Xairete/otus_mlops_final_project/assets/46996898/213f7743-b3bf-444e-8eab-150c40408cf8)
![image](https://github.com/Xairete/otus_mlops_final_project/assets/46996898/cff2fb4a-f1d0-4ac2-8bd5-5b5607533e2d)
![image](https://github.com/Xairete/otus_mlops_final_project/assets/46996898/a501ce12-9113-4879-aec7-2a40006fed18)
