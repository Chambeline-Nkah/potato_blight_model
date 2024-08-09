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
The simple machine learning model was trained using the basic convolutional neural network architecture with 50 epochs, achieving an accuracy of 99.24% on the test dataset, with a loss of  0.0211 indicating a strong performance, while the accuracy on the validation dataset was 94.53% with a validation loss of 0.1431. This proves that the model adapts well to new, unseen data and is robust in its performance.

Model 2: Model with Optimization Techniques
The second model was trained using Adam as an optimizer and using L1 regularisation still with 50 epochs, achieved an accuracy of 98.39% on the test dataset with a loss of 0.0744, while the accuracy on the validation dataset was 96.88% with a validation loss of 0.1103.  This suggests that the model with optimization techniques performed well on the test dataset but struggled with overfitting on the validation dataset.

The key findings are:
1. The simple machine learning model, trained for 50 epochs, achieved a high accuracy of 99.24% on the test dataset, with a low loss of 0.0211. This suggests the model was able to generalize well to new, unseen data.
2. The model with optimization techniques, including L1 regularization and 50 epochs of training as well, also performed well, achieving 98.39% accuracy on the test dataset. However, it struggled more with overfitting, as evidenced by the lower 96.88% accuracy on the validation dataset.
3. The data preprocessing steps, including scaling the pixel values and applying data augmentation, were crucial in improving the models' performance and preventing overfitting.

In conclusion, both models performed well, with the simple machine learning model demonstrating particularly strong and robust performance.


Now, inorder to have everything running locally on your PC, you may follow the following steps:
- Clone the repository: git clone <repository_url>
- Navigate to the potato_blight_model folder: cd potato_blight_model

1. Inorder to run the FastAPI, follow the following steps:
- Navigate to the api folder: cd api
- Run the main.py file using python on your terminal: python main.py
- Then you can make a post request

2. Inorder to run the frontend application, just execute the following:
- Navigate to the potato_blight_model folder: cd potato_blight_model
- Navigate to the frontend folder: cd frontend
- Run the following command on your terminal: npm run start
- When the UI screen opens, select any random image of a potato leaf locally saved on your PC and it will print out the name of the disease the potato plant is suffering from, as well as the confidence level.
