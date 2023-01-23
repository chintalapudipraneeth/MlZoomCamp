# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:49:41 2022

@author: USUARIO
"""

import requests


url = 'http://localhost:9696/classify'

data = {'url': 'https://github.com/carrionalfredo/Capstone_1/raw/main/images/Test_images/test_02.jpg'}

result = requests.post(url, json=data).json()

print('Kirmizi: ', result.get('Kirmizi_Pistachio')*100,'%')
print('Siirt: ', result.get('Siirt_Pistachio')*100,'%')