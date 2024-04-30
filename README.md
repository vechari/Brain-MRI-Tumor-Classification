# README

Two Convolutional Neural Networks that detect if there is a tumor in a Brain MRI scan, using the Tensorflow Keras API. 
Yes_Or_NO_Detection.py determines if the MRI scan has a tumor, classifying into 2 categories: yes or no. 
Tumor_Type_Detection.py determines if there is a tumor and what type of tumor, classifying into 4 categories: glioma, meningioma, pituitary, and no tumor.

new_images contains two folders contains the dataset for the yes or no classificaiton. tumor_type_images contains the dataset for the type of tumor detection. Pred contains 4 prediction images to test on.

The yes or no detection model reached an accuracy of 97.67%. The type of tumor detection reached an accuracy of 91.10%.
