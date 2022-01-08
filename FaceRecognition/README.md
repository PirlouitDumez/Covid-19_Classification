## Presentation of the project
### Presentation of the proposal
The aim of this project is to develop a facial recognition login system.  The main objective was to create everything ourselves, starting from scratch. We developed the architecture of the detection and facial recognition models, which we then trained. 

### Functional scope
First of all, we integrated the possibility for a user to create a new account. The user can create an account by entering their email address and login. The new user is then photographed with their webcam as they enter their information, and the photos are saved so that they can log in later using facial recognition.

Then we added a secure facial recognition login system. The user has to enter their ID, and the system checks if the person behind the camera is the one trying to log in. 


### Technical choices (hardware and software)
Below is a list of the hardware and software used to complete our project.

- We developed everything in python, for compatibility reasons between the different parts.
- For facial recognition, we used the Keras and Tensorflow libraries to create the different neural networks and train them. We used OpenCV to manipulate the images, and many other modules such as Numpy for the tables, or MatPlotLib for the different graphics.


## Functional analysis
### Functionality details
Our final solution includes the following features.

- The user enters his ID. The face is then compared to the ID to detect if it matches or not. If it is a match, the user is redirected to the home page of the website.
- A new user can create an account on the website. All he has to do is to enter his e-mail address and the username he chooses. In order to test our solution more easily, we have removed the database that allowed the account creation to be secured. Indeed, the administrator had to authorise a user to register, instead of giving a special identifier.


### Mock-ups
Here are the mock-ups of the solution we have developed.


<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/Aspose.Words.e148c4a0-0357-478c-95de-9f103658d36e.002.png>

*Account registration page*

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/Aspose.Words.e148c4a0-0357-478c-95de-9f103658d36e.003.png>

*Login page*


##
## Facial recognition implementation

Artificial Intelligence is a science in which we try to enable a machine to reproduce tasks that can be done by humans. Artificial Intelligence consists of several areas, one of the main ones being Machine Learning.

Machine Learning is defined by one of its pioneers Arthur Samuel as a discipline that aims to develop algorithms that give a computer the ability to learn, rather than programming it explicitly.

DeepLearning is a sub-category of Machine Learning in which neural networks are used. Originally inspired by biological neurons, neural network algorithms are capable of performing any task. 


For example, a neural network may be able to play chess, classify different images, or recognise a handwritten number. The applications are vast, and one of them is Computer Vision, which consists of "giving eyes" to a computer, i.e. teaching it to see and interpret the content of an image. In the context of our project, our objective is to teach a programme to recognise a person in an image, so it is the principle of Computer Vision that we are using. 

To do this, we decided to create the architecture of our different models ourselves. In Machine Learning, a model is an algorithm that learns to perform a specific task from a set of data, called a dataset. The objective is to give the model a training set (trainset), and give it the answer it should output, so that it can learn to give the correct output. Then, we test the model on a testset, to see if it is able to predict an output knowing only the input. 

The dataset is a fundamental part of Machine Learning, because the machine learns with a lot of data, and the better the data, the better it can learn. So we decided to create our dataset ourselves, by taking pictures with the webcam of many people from ISEN, and we also retrieved datasets from the Internet to have more data. We used the LFW dataset (Labeled Faces in the Wild) and the CelebA.

The complexity of our project lies in the fact that we create everything ourselves, from the architecture of our models to the training of the models, including the creation of part of the training dataset.

*Sample of the datasets used*.



|**LFW**|**CelebA**|
| - | - |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/LFW.png width=200 height=200> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/celebA.png width=200 height=200> |


Facial Recognition can be broken down into two exercises. First of all, it is necessary to extract the faces present on an image. This is the detection phase. Then comes the recognition phase, in which the goal is to determine to whom the extracted face belongs. To do this, we create what is called a pipeline. A pipeline is, in deep-learning, a series of transformations to which an image is subjected. 

Let's look in detail at the recognition and detection models.



### Detection models
###### First proposal
The aim of the detection model is to extract the different heads present in a photo. For this purpose, we have tested different convolutional neural network (cnn) models. Convolutional neural networks are a type of neural network particularly adapted to images. It works by applying successive filters on the input image.

The first method we implemented is based on the sliding window principle. The detection task is divided into two simpler tasks:

- Extraction of areas from the image
- Classification of these areas into two categories: head or no head

For the first part, we use a sliding window: The whole image is scanned through a smaller window in order to extract enough areas to send for classification. 

However, with this method, one and the same head was detected several times by our algorithm, creating overlaps of proposals as in the figure below. It was then necessary to implement a non-maximum deletion algorithm to keep only the best regions. If a region overlaps another region more than a certain threshold (e.g. 30%), only the best rated region is kept.

|**Before nonMaxSuppression**|**After nonMaxSuppression**|
| - | - |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms1.png width=300 height=300> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms1.png width=300 height=300> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms2.png width=300 height=350> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms2.png width=300 height=350> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms3.png width=300 height=300> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms3.png width=300 height=300> |


However, we did not choose this sliding window solution because of its slowness. Indeed, it takes between 1.2 and 1.4 seconds to detect the images on the photo. Such a slowness is explained in particular by the numerous sub-images to classify (it takes more than 0.5s to classify the 324 sub-images here), and the non-max suppression algorithm, which takes between 0.1 and 0.15 seconds.

###### *FACE CLASSIFIER*

This is the architecture used for the classification model. The principle is to progressively reduce the dimension of the image through layers of max-Pooling and layers of convolutions. Then the actual decision is made by the three dense layers. The two perceptrons in the last layer are each worth the probability that the image is a head or not.
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model0.png>


###### *SLIDING WINDOWS*

In summary, the principle of sliding windows is explained below: A window goes through the image to split it into sub-images. This set of images is then given to the classifier and then to the non-max-suppressor. In this way it is possible to determine the location of a head.
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/pika1.png>

##### ** Second proposal.

The second solution chosen was to create a model - called a Region Proposal Network (RPN) - that determines the location of heads in an image in a single run. 

The original idea was to train a model to propose regions of interest that might contain a head, and then send this data to the classifier, thereby reducing the number of sub-images from 324 to a mere ten. However, the model proved to be more efficient than expected, so we were able to remove the classifier behind and keep only this RPN. We trained this model for four full days on the workstation of the ISEN Info club, on a lot of data (more than 25000 images). This quantity of data allowed us to avoid overfitting (i.e. overlearning). 

Indeed, when a model is too complex or is not trained on a lot of data, it tends to learn the training data too precisely, fitting it perfectly, and is therefore no longer able to generalise on the test data that it has never seen before.

In our case, the training data contained, as input, a photo dataset of people, and as output a mask of the heads, i.e. a black image with a white square at the location of the head. Our RPN learned to extract such a mask from any image.

A quick processing of the model output (edge detection, boundary coordinates recovery) then allows to extract precisely the heads from the image.

As this method is more accurate and faster (it takes 0.14 to 0.17s to process an image), it is the one we decided to keep.



|*Image supplied to the model*|*Output from the model*|*Output processing*|
| :-: | :-: | :-: |







###### *REGION PROPOSAL NETWORK*

This network uses a U-net type architecture. This means that it will progressively decrease the size of the image thanks to max-pooling, and then increase the size of the image in stages. The specificity of this network is that it has residual connections between the encoded images and the decoded images of the same size. These connections ensure that no information is lost through image compression and that well-defined boundaries are obtained. 
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model1.png>
### Recognition models
As for the recognition part, we also experimented with several models before obtaining a sufficiently powerful model.

Face recognition relies heavily on a model - which we will call the encoder - that learns to transform an image of a face into a vector representative of that face.

If the encoder is working well, it is sufficient to calculate the Euclidean distance between our vector and the vectors of the faces whose identities we know. The distance will be minimised by the same person. In other words, a person has fairly similar vectors.

The major problem with this recognition principle arises in the training of the encoder. Effectively, this is not supervised learning: we do not know the vectors in advance to train the model on.

###### AUTOENCODER
The first idea we had to train such an encoder was to train an auto-encoder and extract the encoder. An auto-encoder is a model that learns to compress an image into a vector of dimension n, and then decompress that image back into the original. It is trained with an image as input and the same image as output. This is the specificity of the model.  

Thus, such an auto-encoder learns to transform an image into a sufficiently significant vector to be able to reconstruct the image with. An intuitive idea of this vector is that it corresponds to a list of specific features (eye colour, jaw shape, glasses or not, etc.).  

We use images of size 128*128*3 as input and a vector of size 100 in the centre, so there is compression by a factor of 491. This architecture allows us to build and train an encoder that transforms an image into a meaningful vector. It is natural to think that the same person has vectors encoded using this model that are quite similar  
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model2.png>  
However, the encoded vector will not be specific to the recognition task and pays too much attention to information not needed for detection but important for image reconstruction, such as face orientation. Indeed, the orientation of the face is important to be able to reconstruct the face faithfully, but not necessary for face recognition.  

Let's see some examples of the output of the trained autoencoder. 

*Here, the latent vector (i.e. the central vector) has a dimension of 64. The input image is therefore compressed by a factor of 768.

|**Autoencoder input**|**Autoencoder output**|
| - | - |


Compression strongly affects the quality of the output image, but the general information of the position of the face, the shape of the mouth, the place of the eyes, etc. is preserved.  




###### SIAMESE NETWORK
The second model we implemented is a "siamese network". Its name comes from the two parallel inputs it has. The two input images are encoded with the same encoder, and then the Euclidean distance between the two encoded vectors is calculated. 

In the training phase, either two images of two identical people or two different people are given as input, and the aim is to minimise the error, by predicting small distances for two images of the same person, and conversely large distances when they are two different people. In this phase, the encoder weights are adjusted to minimise the cost function and thus best meet this task.  
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model3.png>  


###### FACENET MODEL

After having tested the siamese network, we decided to use a new architecture to obtain better results, the FaceNet model. The principle of this model is not to have 2 inputs, but this time 3 inputs. An anchor input, i.e. an image of a randomly chosen person which is our reference image, then an image of the same person (the so-called "positive" input), and an image of a different person (the "negative" input). 

As with the Siamese network, we use the same encoder to encode the different input images. We then calculate the error of our model using a "triplet loss". This cost function gives 0 if the distance between the anchor and the positive is sufficiently smaller compared to the distance between the anchor and the negative. Thus, during training, the model learns to predict large distances when the individuals are different, and small distances otherwise. The strength of this model lies in the fact that it minimises the distance between two identical people at the same time as it maximises the distance between two different people.

This model is trained with three images as input, and 0 as output, regardless of the images.  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model4.png>  

`           	`In order to optimise the performance of our model, we tested several encoder architectures on our FaceNet model. The encoder we finally chose was the "beta encoder", which gave us the best results in terms of accuracy (the results for each model are shown in the summary table) and speed.

The architectures we tried are described below.


###### ALPHA ENCODER
This encoder is quite promising although slightly overfitting. Factoring the conv 5\*5 layer into two conv 5\*1 and 1\*5 layers greatly reduces the number of parameters (10 instead of 25), and therefore the training speed.  
###### 
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model5.png>  


###### XCEPTION ENCODER
Training a complex model on a lot of data takes a lot of time. This is why we tried to use a pre-trained model (the Xception model from google). The model was trained on an image classification task. We took this model and its parameter values, and then "froze" half of the model and trained the other half (starting from the pre-trained weights).

This model has heavily overfitted the dataset because there are too many parameters (nearly 8 million trained parameters and 14 million untrainable parameters). Moreover, the performance is disappointing because the task on which the Xception was trained is too far from our encoding task. 

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model6.png>  


###### BÊTA ENCODER
The best results are obtained with this encoder. Separating the input data from the beta block provides a different view of the input. The three convolution outputs are concatenated. This ensures that no data is lost, but increases the number of parameters. To compensate for this addition of parameters, a Batch normalisation layer and a Dropout layer (to limit overfitting) are used.  
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model7.png>  




###### ENCODER + SKLEARN CLASSIFIER
In order to improve the performance of our model, we tried to add a classification model from the Scikit-Learn library (a library that offers many machine learning models) to the output of the encoder. Indeed, the Euclidean distance (as well as the cosine similarity) gives equal importance to all the values of the encoded vector of the image. We therefore assumed that doing a weighted distance could increase the encoding results. We therefore used a randomForestClassifier (to take advantage of its non-linearity) with 100 estimators - a value for which we had a good compromise between bias and variance. We trained it to determine whether two vectors correspond to the same person, based on their difference. With this method, we obtained the best score on the test set by reaching 91% accuracy.

However, the model was computationally heavy and therefore too long to implement on the site, so we finally kept the Beta architecture, in order to have a good accuracy-fastness compromise.
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model8.png>




###### RESULTS

The models presented above give the following results.

The performance (accuracy) of the models is calculated as follows: 

y= y>treshold N×100

y :True value 0 or 1

` `y :Predicted value (float) 

` `treshold :Chosen value (float) 


|**Models**|**Accuracy (%)**|**Treshold**|
| - | - | - |
|**autoencoder**|63.69|1.58|
|**siamese**|64.60|0.31|
|**faceNet Xception**|68.29|3.43|
|**faceNet alpha**|69.96|1.10|
|**faceNet simple encoder**|71.39|3.93|
|**faceNet bêta**|77.25|2.03	|
|**encoder+sklearn**|91.33|-|

It is interesting to note the impact of the architecture of a model on its performance. This is why we tried so many different models, and ended up with the remarkable performance of the encoder+scikit-learn model.


### VISUALIZATION

In concrete terms, what does the encoder do? The encoder assigns a representative vector to each face. We therefore have "groups" of points corresponding to different people. It is possible to visualise these groups by reducing the dimension of the encoded vector to 2, and thus displaying the points on a plane. To reduce the dimension, we use a PCA (principal component analisis) of scikit-learn.

Each point corresponds to a face image, and each colour to a person. We notice that the vectors corresponding to the same person are quite close. The use of the Euclidean distance to compare two persons then makes sense.
<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/viz1.png>


### CONCLUSION

Our system is therefore based, in conclusion, on a deep-learning pipeline which is shown below.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/pika2.png>  

2020/2021 – ISEN
