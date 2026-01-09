# Ideas for Neural Network

## Idea 1:

#### Idea for modifiable layer sizes : 
Get a list the length of which is the number of all the hidden layers where each index corresponds to each hidden layer in order. 
The indexes of the list will contain an integer depicting the number of neurons in that layer

#### Idea for regularization:
Add a regularization parameter in the network object initialization. Will need to add methods of using it's derivative for back 
propagation. 

#### Idea for more loss functions:
Currently I only have Mean Squared Error (MSE) and I need to make that generalizable to other loss functions in a similar vein to 
how the activation functions and regularization functions are used. Will have to rename the activation functions file into just 
'important functions' I suppose.

#### Cross entropy and batch training:
I have to learn what those mean and how to implement them but I know the general idea of them and I know it'll be good for larger datasets.

