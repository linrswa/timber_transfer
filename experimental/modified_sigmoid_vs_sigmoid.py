#%%
import torch
import math
import matplotlib.pyplot as plt

# Defining the modified sigmoid function
def modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7): 
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold

# Generating a range of values
x_values = torch.linspace(-10, 10, 100)

# Calculating the values for both functions
sigmoid_values = torch.sigmoid(x_values)
modified_sigmoid_values= modified_sigmoid(x_values, exponent=10.0, max_value=2.0, threshold=1e-7)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values.numpy(), sigmoid_values.numpy(), label='torch.sigmoid()')
plt.plot(x_values.numpy(), modified_sigmoid_values.numpy(), label='modified_sigmoid()')
plt.title("Comparison of torch.sigmoid() and modified_sigmoid()")
plt.xlabel("x")
plt.ylabel("Function value")
plt.legend()
plt.show()