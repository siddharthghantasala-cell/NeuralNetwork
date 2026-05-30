# Ideas for Neural Network

#### Idea for modifiable layer sizes : 
Get a list the length of which is the number of all the hidden layers where each index corresponds to each hidden layer in order. 
The indexes of the list will contain an integer depicting the number of neurons in that layer

#### Idea for regularization:
Add a regularization parameter in the network object initialization. Will need to add methods of using it's derivative for back 
propagation. Will probably implement it like how I did with the activation functions.