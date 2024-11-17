# Custom Optical Character Recognition (OCR) System Using a Scratch-Built Convolutional Neural Network

<p align='justify'>
This application leverages a Convolutional Neural Network (CNN) to predict handwritten digits drawn by the user on an interactive canvas. 
Users can draw a number (from 0 to 9) on the canvas, and the CNN model processes the input image in real-time. After submitting the drawing, 
the app converts the canvas into a format suitable for prediction, rescaling and preprocessing the image before passing it through the trained CNN. 
The model then analyzes the features of the drawing—such as edges, shapes, and textures—through its multiple convolutional layers and predicts the number.
</p>

### Key Features:
-  Real-Time Prediction: Upon submission,  the drawing is processed and a prediction is made by the CNN, displaying result instantly
-  Interactive Drawing Canvas: Can freely draw using brush tool, simulating the input for handwritten digit recognition.

### Language and Libraries Used:
-  Language: Python
    -  numpy
    -  pillow(image processing)
    -  tkinter(interactive canvas)
 
### Snipped Images:
-    Frontpage:<p align='center'><img src="Images/homepage.JPG"></p>
-    Setting Hyperparameters:<p align='center'><img src="Images/hyperparameters.JPG"></p>
-    Initialized:<p align='center'><img src="Images/initialized.JPG"></p>
-    Canvas with accuracy:
      -    with correct prediction: <p align='center'><img src="Images/correctpred.JPG"></p>
      -    with wrong prediction: <p align='center'><img src="Images/wrongpred.JPG"></p>

This project is designed as an educational endeavor to deepen my understanding of Convolutional Neural Networks (CNNs) by exploring their architecture, functionality, and practical applications in tasks such as image recognition and classification.
