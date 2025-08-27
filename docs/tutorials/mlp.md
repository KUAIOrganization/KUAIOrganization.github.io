# Making a Multi-Layer Perceptron (MLP)

this is a simplified intro to MLPs, enough to understand how they process structured/tabular data. their most common use case is classification or regression on feature vectors (lists of numbers), like predicting if a student passes an exam based on hours studied and hours slept (this time we can do it over 100s of students rather than 1 guy).

mlps can technically take any 1d array as input — unlike cnn’s, they don’t exploit spatial locality, so neighbors in the input don’t matter as much.

## Step 0: prereqs and theory

an MLP is just a standard feedforward neural network. the input goes through fully connected layers (linear transformations) interleaved with non-linearities.

if you want a deeper understanding, check out [Neural Networks from Scratch](https://nnfs.io/).

we will use pytorch from the start. first, import what we need:

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

## Step 1: preparing the data

suppose we want to predict if a student passes (`1`) or fails (`0`) based on hours studied and hours slept. our dataset could look like this:

```python
X = torch.tensor([
    [5.0, 7.0],
    [1.0, 2.0],
    [10.0, 6.0],
    [2.0, 8.0]
])
y = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])
```

`X` is our input (2 features per student), `y` is the target, now with 4 students.

batch size is optional here, but for more data, wrap it in a `DataLoader` like with cnn’s.

## Step 2: building the MLP

an MLP consists of:

- fully connected (`Linear`) layers — these map inputs to outputs
- activation functions — like ReLU or Sigmoid

The architecture looks like this:

<img width="736" height="348" alt="image" src="https://github.com/user-attachments/assets/d63dcecc-1c35-4017-9172-204a71fab681" />

here’s a minimal 1-hidden-layer MLP:

```python
model = nn.Sequential(
    nn.Linear(2, 4),  # input layer → hidden layer
    nn.ReLU(),        # activation
    nn.Linear(4, 1),  # hidden layer → output layer
    nn.Sigmoid()      # squash into 0–1 probability
)
```

* input layer: 2 features → hidden layer: 4 neurons
* output layer: 1 neuron → probability of passing

you can test it with a dummy input:

```python
dummy = torch.tensor([[3.0, 5.0]])
print(model(dummy))
```

## Step 3: defining loss and optimizer

for binary classification, we can use `BCELoss` (Binary Cross Entropy), basically cross entropy for two choices

```python
loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=0.01)
```

learning rate is higher here because the network is tiny.



## Step 4: the training loop

like cnn’s: forward → loss → backprop → update

```python
for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")
```

after training, the model will output probabilities close to 0 or 1.

## Step 5: saving and loading the model

### saving weights:

```python
torch.save(model.state_dict(), "mlp_student.pth")
print("model saved.")
```

### inferencing later:

```python
# define the same architecture
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# load weights
model.load_state_dict(torch.load("mlp_student.pth"))
model.eval()

# predict new student
new_student = torch.tensor([[6.0, 5.0]])
with torch.no_grad():
    prob = model(new_student).item()
    print("pass probability:", prob)
```

## Step 6: intuition/a way to think about it

* each layer linearly combines inputs via weights/biases → applies non-linearity → passes forward
* hidden layer “learns features” from raw input
* output layer maps to final prediction probability

## Full code (put together)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# data
X = torch.tensor([[5.0, 7.0], [1.0, 2.0], [10.0, 6.0], [2.0, 8.0]])
y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# loss + optimizer
loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=0.01)

# training
for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")

# save
torch.save(model.state_dict(), "mlp_student.pth")
print("model saved.")

# inference
model.load_state_dict(torch.load("mlp_student.pth"))
model.eval()
new_student = torch.tensor([[6.0, 5.0]])
with torch.no_grad():
    print("pass probability:", model(new_student).item())
```
