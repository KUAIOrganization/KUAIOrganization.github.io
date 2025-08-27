# Making a Convolution Neural Network, a CNN

this is a simplified intro to cnn’s, enough to understand how they "see" images, which is there most common usecase.

to those of you who have come from the data shapes section, a CNN will work on any 2d array, but will also perform better if the neighbors in the 2d array are related. An Image is literally a 2d array (every pixel is an rgb list, pixels from a 2d list), and nearby pixels are related (either an object or a boundary), so their most common usecase is image classification, but any 2d array works

The CNN architecture looks like this:

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ea2cbc06-89cd-4902-923e-508e69f5cb64" />

## Step 0: prereqs

This is a Full Pytorch based tutorial, so you have to get used to pytorch data handling
