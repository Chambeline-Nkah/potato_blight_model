Image Classifier Model

Overview of the dataset
The dataset contains images of potato leaves with three common diseases: early blight, late blight, and healthy leaves. The dataset consists of a total of 1,500 images, with 500 images for each class (early blight, late blight, and healthy leaves). The images have a resolution of 256x256 pixels.

Tasks
Implement two models :
A simple machine learning model based on neural networks on the chosen dataset without any optimisation techniques and
A model applying at least three optimisation techniques.

Model architecture
The model was built using a convolutional neural network (CNN) architecture. The model consists of several six convolutional layers, six pooling layers, and fully connected layers. The input to the model is a 256x256 pixel leaf image, and the output is a classification of the image into one of the three classes (early blight, late blight, or healthy).

The model was trained on a dataset of 1200 images and a validation dataset of 150 images. The model was evaluated on a test dataset that contains 150 images.
Data pre-processing
The pixel values of the images were scaled as this helps the model converge faster and perform better by ensuring that the input data is on thesame scale. Again, to ensure better performance, data augmentation was implemented as this aided in increasing the diversity of the input data and preventing overfitting.

Model training and evaluation
The model was trained using the Adam optimizer and a categorical cross-entropy loss function. The model was evaluated on the test dataset using the accuracy metric.

Model 1: Simple Machine Learning Model
The simple machine learning model was trained using the basic convolutional neural network architecture with 50 epochs, achieving an accuracy of 98.81% on the test dataset, with a loss of  0.0394 indicating a strong performance, while the accuracy on the validation dataset was 99.22% with a validation loss of 0.0435. This proves that the model adapts well to new, unseen data and is robust in its performance.

Model 2: Model with Optimization Techniques
The second model was trained using Adam as an optimizer and using L1 regularisation but this time around with only 30 epochs, achieved an accuracy of 97.80% on the test dataset, while the accuracy on the validation dataset was 84.38% with a validation loss of 0.5835.  This suggests that the model with optimization techniques performed well on the test dataset but struggled with overfitting on the validation dataset.
The comparison between the simple model and the model with optimization techniques highlights the importance of the number of epochs in the performance of a model. The simple model used 50 epochs, which resulted in a high accuracy and low loss on both the test and validation datasets. In contrast, the model with optimization techniques used 30 epochs, which resulted in a lower accuracy and higher loss on the validation dataset. This suggests that the number of epochs plays a significant role in the performance of a model, and that using more epochs can lead to better results.
