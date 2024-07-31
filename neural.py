import torch
import torchsummary


#comment the activation
class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(4, 6)
        self.activation1 = torch.nn.LeakyReLU()
        #allows a small, non-zero gradient when the unit is inactive and not in the positive range.
        
        self.linear2 = torch.nn.Linear(6, 8)
        self.activation2 = torch.nn.Tanh()
        #Outputs values in a range between -1 and 1, which helps in centering the data
        
        self.linear3 = torch.nn.Linear(8, 4)
        self.activation3 = torch.nn.Sigmoid()
        #maps the input to a value between 0 and 1, often used for binary classification or in the output layer for probabilities
        
        self.linear4 = torch.nn.Linear(4, 4)
        self.activation4 = torch.nn.ELU()
        #Addresses the vanishing gradient problem and speeds up learning by combining the benefits of ReLU and allowing negative values.
        #Function: Exponentially decays for negative inputs and is linear for positive inputs.
        
        self.linear5 = torch.nn.Linear(4, 2)
        self.activation5 = torch.nn.Softmax(dim=1)
        #converts raw scores into probabilities that sum to 1, typically used for the final layer in multi-class classification problems.       

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)        
        x = self.linear4(x)
        x = self.activation4(x)
        return x

tinymodel = TinyModel()


input_tensor = torch.randn(1, 4)  
#(batch , inputs)

output = tinymodel(input_tensor)

print('Input Tensor:')
print(input_tensor)

print('\nOutput Tensor:')
print(output)

torchsummary.summary(tinymodel, (4,))