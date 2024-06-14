# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
import torch.nn as nn

from config import Config
sys.path.append("..")

class ImageAnalyser:
    def __init__(self, model_weights_path):

        self.classes = ['Не еда', 'Еда']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cpu = torch.device("cpu")
        try:
            model = models.resnet50(pretrained=False)
            model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512),
                                  nn.ReLU(), 
                                  nn.Dropout(), 
                                  nn.Linear(512, 2))
            session = boto3.session.Session(aws_access_key_id = '***', 
                                    aws_secret_access_key='***')

            s3 = session.client(
                    service_name='s3',
                    endpoint_url='https://storage.yandexcloud.net')
            s3.download_file('mlops-dev-german', 'model.pth', model_weights_path)
            model_checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_checkpoint)
            #model.to(self.device)
            model.train(False)
            model = torch.nn.DataParallel(model)

            self.model = model
            img_dimensions = 224

            self.transforms = transforms.Compose([
                transforms.Resize(img_dimensions),
                transforms.CenterCrop(img_dimensions),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
                ])

        except Exception as ex:
            
            raise

    def _predict(self, image):
        image = Image.open(image).convert('RGB')
        if image.size[0] > 500 or image.size[1] > 500:
            return ['Error', 1]
        inputs = self.transforms(image)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs.unsqueeze(0))
            result =  self.classes[output.argmax()]

        return [result, 0]

    def inference(self, image):
            
        self.model.to(self.device)
        result = self._predict(image)
        self.model.to(self.cpu)
        return result


class AppContext(object):
    classifier = ImageAnalyser(Config.MODEL_NAME)


CONTEXT = AppContext()