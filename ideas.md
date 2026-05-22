# Ideas for Neural Network

#### Idea for modifiable layer sizes : 
Get a list the length of which is the number of all the hidden layers where each index corresponds to each hidden layer in order. 
The indexes of the list will contain an integer depicting the number of neurons in that layer

#### Idea for regularization:
Add a regularization parameter in the network object initialization. Will need to add methods of using it's derivative for back 
propagation. Will probably implement it like how I did with the activation functions. 

#### Idea for more loss functions:
Currently I only have Mean Squared Error (MSE) and I need to make that generalizable to other loss functions in a similar vein to 
how the activation functions and regularization functions are used. Will have to rename the activation functions file into just 
'important functions' I suppose.

#### Idea for adding batch dimensions:
Currently, during training, the program loops through all the datapoints in the batch which completely skips the usefulness of the batch. I
need to add a 'batch dimension' to everything that includes batches and completely skip the Layer.update() and Layer.backward() methods
as there will then be no need for looping

#### Idea for exporting and saving models:
It's definitely a little stupid to be training models only to have it get annihilated from this world when Execution.py is finished running.
I feel like I should've definitely added that before or at least have it have a larger priority on the list of todos

