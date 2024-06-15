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
Схема сервиса  
![image](https://github.com/Xairete/MLOps_final/assets/46996898/786ebd22-da8b-4ce8-a9c2-808befa630d8)

Внешний вид интерфейса  
![image](https://github.com/Xairete/MLOps_final/assets/46996898/b4e702e5-5e31-4ba4-bbff-3725481e324c)  

![image](https://github.com/Xairete/MLOps_final/assets/46996898/dccf0298-61f2-4f2c-a9bd-21001394894a)

![image](https://github.com/Xairete/MLOps_final/assets/46996898/e20bf5f6-35ba-41fa-8e35-95b1fd388196)


