# Making a Convolutional Neural Network, a CNN

this is a simplified intro to cnn’s, enough to understand how they "see" images, which is there most common usecase.

to those of you who have come from the data shapes section, a CNN will work on any 2d array, but will also perform better if the neighbors in the 2d array are related. An Image is literally a 2d array (every pixel is an rgb list, pixels from a 2d list), and nearby pixels are related (either an object or a boundary), so their most common usecase is image classification, but any 2d array works

## Step 0: prereqs and theory

This is a convolutional neural network, if you want to learn more about convolutions and how they work, check out [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) by 3b1b

This is a Full Pytorch based tutorial, so you have to get used to pytorch data handling

For this tutorial we will be using the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database), its literally 0-9 written by hand, so its a task about classifying handwritten numbers. 

Its relatively popular, importing it in is simple

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()), batch_size=32, shuffle=True)
```

What does this codeblock do?

First we have our imports:

 - `torch` - Our actual machine learning library, this is where all our functions and classes come from
 - `nn` - technically a part of torch, the most commonly used one, named for convenience
 - `optim` - technically a part of torch, we've named this one for convenience, better than writing `torch.optim` everywhere, used for the optimizer in your training
 - `torchvision` - This is pytorch's vision toolkit, basically any function that helps out with vision can be found here, along with common vision datasets. transforms are a major feature as well, they are transforms for images, you can [Random Crop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html), [Random Flip](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html), or even [Color Jitter](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)
 - `torch.utils.data`, `DataLoader` - relates to file data and checkpoint saving, relevant later

From here we actually load our data

A dataloader is a class designed to stream your data to a model in training.

With a proper machine learning model, you will usually have an 80/10/10 split of the data, known as Train/Val/Test. Train is the data the model actually sees, the data the model learns from. Val is a part of the data the model never sees, but is tested against while training. This helps look into issues like "is my model memorizing part of the data" or "is my model actually learning patterns, and can it solve out of distribution (outside of train) questions?". Test is another part of the data the model never sees, but isnt really tested against until after training and you've picked your model weights. This is to prevent the model from getting to used to the val data, as many professional workflows will use early stopping, or val loss trends, to continue or stop training a model, which might make your model fit to val as well. With test splits you can prevent this issue.

A general tip for data splits is to make sure the splits are uniform, if a split is too different from the other splits, ex. train is 90% blue images but test is 90% red images, comparing the two no longer gives you accurate results, and ruins the point of having splits. Ideally if train is 90% blue and 10% red, val/test should also be 90% blue 10% red.

As this is a beginner tutorial, we will only be using the train split.

`download=True` is there so you actually get the data, and transforms here is just `transforms.ToTensor()`, which is just turning your data into a tensor, a tensor is basically just a "machine learning list", like imagine a python list but for ML, and thats a tensor. Using `transforms.ToTensor()` is mostly the same as using `int(input())`, just making sure its the right format.

here we only have the one transform so its a one-liner, but usually you would do something like this:

```python
your_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256 pixels
    transforms.RandomCrop(224),     # Randomly crop a 224x224 region
    transforms.RandomHorizontalFlip(), # Randomly flip horizontally
    ...
    transforms.ToTensor(),          # Convert PIL Image or NumPy array to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize pixel values
])

train = DataLoader(datasets.MNIST( ..., transform=your_transforms), ...)
```

`batch_size` is just how many samples you are sending to the model at once, usually more is better, keep it going till you CPU/GPU runs out of memory or you no longer see performance gains
`shuffle` randomizes the order, generally an improvement to model training, if you want to keep the order the same use `torch.manual_seed(<any_number>)`

## Step 1: Mapping out a CNN

now that you've got the data loading, we need to design the model that data has been loaded for

The CNN architecture looks like this:

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ea2cbc06-89cd-4902-923e-508e69f5cb64" />

we take the input, apply a convolution, add a non-linearity, and repeat for a couple blocks, before adding the classification block, where we flatten everything into a giant row, and use a linear layer to morph that into our 10 possible choices

## Step 2: The convolution block

Since this is a smaller neural network we wont literally make a block, but with larger neural nets you can define a class `CNN_block` and just call it with `CNN_block()`

our Conv block has 3 parts,
 - taking an input
 - applying a convolution
 - adding a non-linearity

As our model uses the image tensor the entire way, and a convolution may change the tensor, but not change the tensor datatype, we can use a single conv as the intake and transform

A convolution itself is made of kernel_size, stride, and padding. A kernel is basically a matrix, and the kernel size is the size of the matrix. A kernel usually takes a group of values, multiplies it by its values, and returns a value, those values are usually specified outside of machine learning, like a kernel whose values average pixels to blur an image, or specific values to sharpen an image, etc. In a CNN, this kernel is filled with learnable values, values the model changes for itself while training to be any of the patterns mentioned above, and more. Often times in more complicated models, there are multiple stacked kernels, which allow for complex changes that pull out the most important features in a model.

The filter will slide around the image, like below:

<img width="1170" height="849" alt="image" src="https://github.com/user-attachments/assets/24bc966f-e48a-43c2-9724-5e0e8cda354c" />

The stride and padding both affect this slide. The stride affects how much the slide moves by, like in the image below, where if stride is higher, the kernel averages the same pixels less and also shrinks the input more. 

Stride:

<img width="294" height="288" alt="image" src="https://github.com/user-attachments/assets/326291e8-f077-4915-bd05-2c9c0d176b00" />

Padding is useful when you want to keep the input the same size as before, as you've seen in the first image, things like 5x5 -> 3x3 happen

padding adds a ghost border for the conv to use, like effectively doing a 5x5 -> 7x7 -> 5x5

<img width="395" height="449" alt="image" src="https://github.com/user-attachments/assets/88c89a30-3f0f-47e9-9e57-e2d2e7c30c3d" />

In pytorch, you can make a conv with `nn.conv2d(...)`, which defines the learnable filters. These learnable filters are controlled with channels, using `in_channel` and `out_channel`, the first two parameters of the conv2d

channels are like color channels or filters, like for an input of a grayscale image, you have 1 channel, while an RGB image has 3. From here you use them to expand into your learnable filters, while using the third argument, `kernel_size` to control the size of the filter

knowing that we have grayscale images, lets say I want a convolution to go from 1 channel to 8 filters, each a 3x3, I would have code like this: `nn.conv2d(1,8,3)` 
you can also use stride and padding here, like this: `nn.conv2d(1,8,3, stride=..., padding=...)` but beware, this will change your tensor size, so you will need to compensate for that.

we can use a quick `ReLU()` for the nonlinearity, use better functions if you'd like.

From here I'd like to introduce the `nn.MaxPool2d()`, similar to the kernels, this instead picks the maximum value from its range, effectively consolidating the most important features, while also shrinking the input.

To chain conv layers, the next conv's in channels must always match the previous conv's out layers, and generally increasing the models channels by powers of 2 (x2, x4, 8, etc), is best

So far the model looks like this (an nn.Sequential stores multiple components to make 1 big network):

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2,2),
)
```

## Step 3: Flattening and classifying

This step is relatively simple compared to the last one, we now only need to flatten the layer and make a linear layer from this flattened array to our choices. A linear layer is great at mapping linear inputs to outputs, but needs that flat layer first.

add an `nn.flatten()` to our nn.Sequential. To get the output, just print the shape from a dummy input, a quick test trick, like this:

```python

...model...

dummy_input = torch.randn(1, 1, 28, 28)

print(model(dummy_input).shape)
```

knowing the output, make a linear layer from that value to 10, the number of choices it can be.

`nn.Linear(16*5*5, 10)  # 10 digits`

Now heres the full model:

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(16*5*5, 10)  # 10 digits
)
```

## step 4: the training loop

from here we add a loss and an optimizer, using cross entropy loss, this is basically the loss you use whenever you have multiple choices, like our 10 classes, compared to something like MSE for accuracy

```python
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
```

> I encourage you to try out different optimizers, and maybe try learning the very basics, as thats math you encounter from calc 1 to calc 3, the rules in calc 1-3 can be used to make backprop and parts of optimizers

from here we define a basic training loop, 1 epoch, predicting, measuring loss, zero-ing your gradients (resets gradients), and performing backpropogation, then repeating the step again

```python
# training loop
for epoch in range(1):
    for X, y in train:
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("epoch done, loss:", loss.item())
```

## conclusion

The full code:

```python
import torch.optim as optim

# dataset: MNIST digits (images + labels)
from torchvision import datasets, transforms
train = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=32, shuffle=True)

# model + loss + optimizer
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(16*5*5, 10)
)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# training
for epoch in range(1):
    for X, y in train:
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("epoch done, loss:", loss.item())
```

The code should take around 30s to train the full model, reaching a loss of around 0.09

If you'd like to learn about maximising performance on this dataset and more modern methods that work on this dataset, take a look at [Deep Neural Nets: 33 years ago and 33 years from now](https://karpathy.github.io/2022/03/14/lecun1989/)

## extra: saving and sharing weights

as your model starts to get more complex and more useful, you'll want to be able to save and share the model, to use in an app/website, or measure the performance of different versions.

Saving a model looks like this, add it after the training loop:

```python
torch.save(model.state_dict(), "cnn_mnist.pth")
print("model saved.")
```

That saves the model `cnn_mnist` in the current directory, as a pickle file, which is like an exe in the way that its stored as binary, but its basically pythons way to save information as a binary.

inferencing, aka using the model, looks like this:

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# define the same architecture
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(16*5*5, 10)
)

# load weights
model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()  # set to inference mode

# preprocessing: same as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# load an image (example: "digit.png"), should be the same size as the dataset, 28x28
img = Image.open("digit.png")
x = transform(img).unsqueeze(0)  # add batch dim

# predict
with torch.no_grad():
    logits = model(x) # logits is basically the probabilty of each choice
    pred_class = torch.argmax(logits, dim=1).item()

print("predicted digit:", pred_class)
```
