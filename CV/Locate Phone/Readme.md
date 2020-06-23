## Locate phone in an image and plot bounding box

Given a small training set of floor images containing a phone, the goal is to find the normalized location of the phone in a new test image.

# Training the model (TrainScript_phone.py)

Since the training set size is very small, image patches have been cropped to create a training set. Additionally, data augmentation has been applied to create 
a larger training set. A CNN based classifier was trained on the image patches after splitting into training and validation sets. Dropout layers were added as well 
to prevent overfitting.

# Evaluating the model (EvaluateScript_phone.py)

Given a test image, divide the image into patches and predict if the phone is present or not using the trained model. Next, create a detection map of the entire image
and find the largest connected region whose area is below a certain maximum area, to prevent false positives. Then draw a rectangle around this countour region, 
defining the bounding box. The mid-point of the bounding box, scaled by the image dimensions, gives the predicted position of the phone in the test image.
