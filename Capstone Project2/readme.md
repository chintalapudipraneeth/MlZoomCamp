# Machine Learning Zoomcamp 2022
## Capstone 1 - Pistachio Image Classifier

The present project was elaborated as the Capstone 1 project, for the [Machine Learning Zooncamp 2022](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) realized by [Datacamp.club](https://datatalks.club/).

The objective of this project, is to develop an machine learning model for classificate images of pistachios, and determine to wich variety belongs.

For this, I propose to build and train a deep learning model, taking a dataset of pistachios varieties as training data for the model.

## Pistachios

<p align="center">
  <img width="640" height="173" src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/640px-Pistachio_vera.jpg">
</p>

[From THOR - Pistachio, CC BY 2.0](https://commons.wikimedia.org/w/index.php?curid=40606682)

The pistachio (Pistacia vera), a member of the cashew family, is a small tree originating from Central Asia and the Middle East. The tree produces seeds that are widely consumed as food.

### Pistachio Varieties

They are classified according to their origin; their colour and size differentiates them. The main types of pistachio are the pistachio of Sicily, which is green, the smaller Tunis and the Pistachio of Levante, yellow and therefore less appreciated commercially.

The main types found in the market are the Noble or Sicily pistachio, with a green and very appreciated almond; Tunis, smaller but equally appreciated, and the Levante, with a yellow edible part and less accepted because of its flavour.

Usually, the pistachio varieties are classified according to their place of origin or culture. Each country has its own selections, whose main differences are the colour, flavour, size, period of harvesting and qualities.

**Some varieties of pistachios**

" Kerman"
Pistachio nut of great size and good quality. Selected in Iran, it was introduced in the U.S.A. and it is also cultivated in Spain (in Castilla-La Mancha) where the fruit ripens during the first fortnight of September.

" Peter"
It is used as a male cultivar with Kerman it has a good polen production and they partly coincide during the flowering period. Selected in California.

" Uzun"
Pistachio nut of average size, long and clear green. It is cultivated in Turkey.

" Kirmizi ‘
Pistachio nut of average size and reddish colour. Along with the cultivar Uzum, it is the most cultivated variety in Turkey.

" Abiad miwahi ‘
Pistachio nut of average size, white colour and excellent quality. Cultivated in Turkey.

" Achouri ‘
Pistachio nut of average size, red colour, excellent quality and very productive. Cultivated in Syria.

" Batouri ‘
Thick fruit of whitish colour and good quality. Important cultivar in Syria.

" Sefideh-Montaz" and " Imperiale de Dameghan"
The fruit of these varieties is round, thick and yellowish. Very appreciated in Iran.

" Kouchka ‘
Quite thick pistachio, cream white colour and good quality.

" Mateur"
Long fruit, average size, yellow greenish colour and good taste quality. It was selected in Tunis and it gives good results in Spain. In Castilla-La Mancha it ripens at the end of August.

" Larnaka ‘
Average size pistachio, less long than ‘Mateur ". Original from Cyprus. It is cultivated in Greece and in Spain, giving good results.

" Aegina ‘
Medium size fruit, long and similar to " Mateur ". It comes from Greece and it also gives good results in Spain.

[Source](https://www.frutas-hortalizas.com/Fruits/Types-varieties-Pistachio-nut.html)

## Dataset

The dataset used in this project is a dataset of 2148 600x600px jpeg images of pistachios, 1232 of Kirmizi type and 916 of Siirt type. This dataset was obtained from [Visualdata.io](https://visualdata.io), and can be found in this [link](https://visualdata.io/discovery/dataset/906f860910230c325f1fa63da88f6c847a06724a).

The dataset used in thsi project can be downloaded form thsi [link](https://github.com/carrionalfredo/Capstone_1/blob/main/dataset/Pistachio_image_Dataset.zip).

![Dataset source](https://www.mdpi.com/electronics/electronics-11-00981/article_deploy/html/images/electronics-11-00981-g001.png)
[source: Visualdata.io](https://visualdata.io/discovery/dataset/906f860910230c325f1fa63da88f6c847a06724a)

The dataset is organized of the following way:
```
Pistachio_Image_Dataset/Pistachio_Image_Dataset
├── Kirmizi_Pistachio
└── Siirt_Pistachio
```

<figure>
  <img
  src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/kimizi_images_dataset.jpg"
  alt="Kimizi images in dataset folder."
  title="Kimizi images dataset">
  <figcaption>Kimizi images dataset</figcaption>
</figure>


<figure>
  <img
  src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/siirt_images_dataset.jpg"
  alt="Siirt images in dataset folder."
  title="Siirt images dataset">
  <figcaption>Siirt images dataset</figcaption>
</figure>

## Preparation of the dataset

Load the images dataset from the  ```Pistachio_Image_Dataset``` folder, and build the train and validation subsets with this parameters: ```batch_size = 32```, ```img_height = 150```, and ```img_width = 150```, through the ```image_dataset_from_directory``` Keras utility, with ```validation_split=0.2```. The results are:

```
Found 2148 files belonging to 2 classes.
Using 1719 files for training.
Using 429 files for validation.
```

The classes of the dataset are:
```
['Kirmizi_Pistachio', 'Siirt_Pistachio']
````

## Build of the base model

The base model consist of:
- Rescaling layer
- Conv2D layer.
- MaxPool2D hidden layer.
- Dropout hidden layer.
- Flatten hidden layer.
- Dense hidden layer.
- Dense output layer.

Defined by:
```python
model = Sequential(name = name)
    
model.add(Rescaling(1./255))
    
model.add(Conv2D(32,3,3, input_shape = (150,150,3), activation = 'relu'))
    
model.add(MaxPool2D(2,2))
    
model.add(Dropout(droprate))
    
model.add(Flatten())
    
model.add(Dense(32, activation='relu'))
    
model.add(Dense(2, activation='softmax', name = 'output'))
    
model.compile(
  optimizer = keras.optimizers.Adam(
    learning_rate = learning_rate
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
    )
```

For compilation of the model, is used ``Adam`` optimizer and the loss function is ``SparseCategoricalCrossentropy``.

<p align="center">
  <img alt="Base Model Architecture" src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/base_model.png">
</p>

## Training of the model

### Training of the base model

The base model was trained with the following hyperparameters:
```python
droprate = 0.5
learning_rate=0.001
epochs = 100
```
The training & validation accuracy and loss values obtained are the following:

<p align="center">
  <img width="600" height="400" alt="Base Model Results" src="https://github.com/carrionalfredo/Capstone_1/raw/main/images/Base_model_results.png">
</p>

### Hyperparameters tuning

In order to improve the accuracy and reduce the loss values, the parameters ```learning_rate``` and ```droprate``` were tuned. After that process, the final model parameters are:
- ```learning_rate = 0.0001```.
- ```droprate = 0.6```.

Also, the ```epochs``` parameter was set equals ```60```.

The training & validation accuracy and loss values obtained for the final model are the following:

<p align="center">
  <img width="600" height="400" alt="Final Model Results" src="https://github.com/carrionalfredo/Capstone_1/raw/main/images/Final_model_results.png">
<p>

Finally, the summary of this final model is next.

````
Model: "Final_Model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)     (None, 150, 150, 3)       0         
                                                                 
 conv2d (Conv2D)           (None, 50, 50, 32)        896       
                                                                 
 max_pooling2d (MaxPooling  (None, 25, 25, 32)       0         
 2D)                                                             
                                                                 
 dropout (Dropout)         (None, 25, 25, 32)        0         
                                                                 
 flatten (Flatten)         (None, 20000)             0         
                                                                 
 dense (Dense)             (None, 32)                640032    
                                                                 
 output (Dense)              (None, 2)                 66        
                                                                 
=================================================================
Total params: 640,994
Trainable params: 640,994
Non-trainable params: 0
_________________________________________________________________
````
## Dependency and Environment Managenent

The trained model and its training logic has been exported to the [```train.py```](https://github.com/carrionalfredo/Capstone_1/raw/main/train.py) script, that generates the best performance ```.h5``` file.

The file [```predict.py```](https://github.com/carrionalfredo/Capstone_1/raw/main/predict.py) loads (in this case) the [```CS107_0995.h5```](https://github.com/carrionalfredo/Capstone_1/raw/main/CS107_0.995.h5) model file and deploy it via web service with **Flask**.

All the dependencies and the virtual environment used in this project are provided in the [```pipfile```](https://github.com/carrionalfredo/Capstone_1/raw/main/Pipfile) uploaded in this repository.

In order to install this dependencies and virtual environment, with ```Pipenv``` installed and once downloaded the ```pipfile``` and ```pipfile.lock``` files in the working directory, execute the next command:

```
pipenv install
```

This will install the dependencies from the ```pipfile.lock``` file. To activate the virtual environment for this project, run:

```
pipenv shell
```
Also, its posible run a command inside this virtual environment with:

```
pipenv run
```

Once activated the virtual environment, the model can be deployed via web service running the following command:
```
python predict.py
```

This will serve the ```pistachio-classifier``` Flask app in the port ```9696```.

To verify that the classifier is working, use the [```test.py```](https://github.com/carrionalfredo/Capstone_1/raw/main/test.py) script. In another command window, go to the working directory, and run:
 ```
 test.py
 ```
 
If all is working OK, in the virtual environment command window, should return a ``"POST /classify HTTP/1.1" 200 -`` message, and in the another command window, should show the results of the prediction.

For this example, the ```test.py``` script download and load into Keras the following pistachio image for classify it.

<p align="center">
  <img src="https://github.com/carrionalfredo/Capstone_1/raw/main/images/Test_images/test_02.jpg" alt="Pistachio for Test" width="200"/>
<p>

If all is working OK, the result of the classification should be similar to:

```
Kirmizi:  85.0 %
Siirt:  15.0 %
```

Indicating that the pistachio image is 87% of Kimrizi variety.

## Containerization

The model and dependencies were containerizated with **Docker**.

To create a Docker image denominated `cs1` with the virtual environment and dependencies used in the model, start the Docker service, go to the working directory where the necesary [`Dockerfile`](https://github.com/carrionalfredo/Capstone_1/raw/main/Dockerfile) is, and run the following command:

```
docker build -t cs1 .
```

The `Dockerfile` used to create the Docker image in this project, has been uploaded to this repository.

To run the Docker image recently created, run this command:
```
docker run -it --rm --entrypoint=bash cs1
```

And for run the web service via **Gunicorn** of the ```pistachio-classifier``` app in the port `9696`, run the following command:

```
docker run -it --rm -p 9696:9696 cs1
```

The following messages should be show:

```
[1] [INFO] Starting gunicorn 20.1.0
[1] [INFO] Listening at: http://0.0.0.0:9696 (1)
[1] [INFO] Using worker: sync
[8] [INFO] Booting worker with pid: 8
```

After that, to test the deployed model, in another command window, run the `test.py` script.

The result of the pistachio image classification ```Kirmizi:  85.0 %
Siirt:  15.0 %```, should be show as response, indicating that the conteinarized service is runnning and working OK.

## Cloud Deployment

Adicionally, the model was deployed in the cloud through **AWS Elastic Beanstalk**. For this, first an application called `cs1_classify` was created under the Docker platform, with the following command:

        eb init -p docker -r us-east-1 cs1_classify

After create the application, con be tested locally executing:

        eb local run --port 9696

If the application in online, the following message will appear.

![](https://github.com/carrionalfredo/Capstone_1/raw/main/images/EB_local_run.jpg)

In another command window, the application can be tested, executing the `test.py` script:

And the result should be:

![](https://github.com/carrionalfredo/Capstone_1/raw/main/images/EB_test.jpg)

For deploy the web service in the cloud, the `cs1-env` environment was created with:

        eb create cs1-env

With the application addres provided by AWS, another test script was created ([`cloud_test.py`](https://github.com/carrionalfredo/Capstone_1/raw/main/cloud_test.py)).

Running the `cloud_test.py` script in a command window, will return the prediction using the `cs1_env` environment and `cs1_classify` application created in the AWS cloud.

![](https://github.com/carrionalfredo/Capstone_1/raw/main/images/EB_cloud_test.jpg)