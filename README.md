# Machine-Learning-Based-Prediction-of-Anti-Aging-Drug-Combinations
This project aimed to identify promising anti-aging drug combinations using generative 
models. To achieve this, two models were trained on the CMap dataset to predict drug
induced transcriptomic profiles: a baseline model and a VAE model (Figure 1). Their performance to predict drug combinations was evaluated using EDs(External Datasets) containing ground-truth 
values. Predicted transcriptomic profiles of 8,001 drug combinations were generated and fed 
into the age model to identify candidate anti-aging and reprogramming drug combinations. 

![image](https://github.com/user-attachments/assets/58a7c66f-b485-41fc-bee1-163d5af1fc26)
Fig 1. Project workflow

Various architecture and hyperparameters were tested to improve the VAE 
model performance, including changes in the number of nodes, layers, dropout rate, activation 
function, learning rate, epochs, and batch size. 
Different combinations of these parameters were tested throughout the experimentation process to identify our final model with the 
lowest validation loss. Figure 2 shows the performance of the VAE model during training and validation. As 
expected, the training loss decreased over time indicating the increasing ability of the model 
to predict data from the training set. The validation loss indicates how well the model 
generalize to new data. The validation loss was initially high and after few epochs became 
identical to the training loss. Such behaviour is desirable and indicate that the model is neither 
overfitting (i.e., the model learns too complex patterns from the training data that do not 
generalize well to other datasets) nor underfitting (i.e., the model is not learning patterns deep 
enough from the training data to derive generalizable conclusions), and that therefore training 
should be continued. After 40 epochs it was observed an increase in the value of validation 
loss compared to training loss, indicative of overfitting, and therefore the further training was 
stopped.

![image]()
