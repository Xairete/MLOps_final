FROM python:3.9.7

ADD requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
RUN pip install gunicorn

ADD app.py /var/server/app.py
ADD config.py /var/server/config.py
ADD models /var/server/models
ADD src /var/server/src
ADD static /var/server/static
ADD templates /var/server/templates
WORKDIR /var/server

CMD python /var/server/app.py

EXPOSE 5000
EXPOSE 5099
