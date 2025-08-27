# Tutorials

## A general intro to machine learning 

_getting the basics down_ -> [Making your first neural network](tutorials/nn_base.md)

## Looking at your problem as a shape

_quickly turning your problem into a model_

We've come across a variety of people who have a problem they would like to solve that they think AI can help with.

A couple examples of this are:

 - estimating the weight of a fish based off of a screenshot
 - optimizing a power grid to predict what areas will need more power in the next few hours
 - attempting to solve 2d puzzles or other arcade games

Most people come in with the assumption that they will need a completely custom neural network and a lot of knowledge to make the smallest of demos, but in reality, most problems can fit within the scope of common model types.

### Start with representing the problem

Think of how you can represent your data, what shapes seem the most natural or obvious, then consult the table below

| model type   | what the input looks like                           | how you usually prepare the data            |
|--------------|-----------------------------------------------------|---------------------------------------------|
| mlp          | a simple list of numbers (like `[5, 7, 3]`)         | put everything into a list, scale values    |
| cnn          | a grid/array of values (like a table or an image)   | arrange data into rows/columns, add depth if needed |
| rnn/lstm/gru | a sequence of steps (like words in a sentence, or time series) | make all sequences the same length, turn words/items into numbers |
| transformer  | a sequence with positions (like a sentence where order matters) | same as above, but also give the model info about position/order |
| autoencoder  | anything, but input and output have the same shape  | clean/normalize data so it can be reconstructed |
| gnn          | a set of points with connections (a network/graph)  | describe which points are linked and what each point's values are |

Links to each NN **(WIP)**:

- [mlp tutorial](tutorials/mlp.md)  **(WIP)**
- [cnn tutorial](tutorials/cnn.md)  **(WIP)**
- [rnn tutorial](tutorials/rnn.md)  **(WIP)**
- [transformer tutorial](tutorials/transformers.md)  **(WIP)**
- [autoencoder tutorial](tutorials/autoencoders.md)  **(WIP)**
- [gnn tutorial](tutorials/gnn.md)  **(WIP)**