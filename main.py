"""
Multiple Linear Regression Model

This program trains a model to predict a patient's blood pressure based on their age and weight.

Author: Davy Tregellas  
Date: 01 May 2025  

Assignment 2: Machine Learning and AI (Task 1)

Code References:

Shah, J. P. (2021). *Multiple Linear Regression from Scratch*. [Kaggle Notebook].  
Available at: https://www.kaggle.com/code/jaypradipshah/multiple-linear-regression-from-scratch/notebook (Accessed: 01/05/2025)

Durfee, C. (2018). *MultipleLinearRegressionPython*.  
Available at: https://github.com/chardur/MultipleLinearRegressionPython (Accessed: 01/05/2025)
"""

# Import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the dataset
dataset = pd.read_csv(r'./data/HousePrices20.csv', sep='\t') # Use tab delimiter since the dataset is separated by tabs

original_sqft = dataset['SqFt'].values
original_bed = dataset['Bedrooms'].values
original_price = dataset['Price'].values

# Store mean and std for inverse scaling
sqft_mean = original_sqft.mean()
sqft_std = original_sqft.std()

bed_mean = original_bed.mean()
bed_std = original_bed.std()

price_mean = original_price.mean()
price_std = original_price.std()

# Check if the dataset loaded correctly by printing first few rows
# print(dataset.head())

# Separating the independent and dependent features

# Dependent variable (target)
y = np.asarray(dataset['Price'].values.tolist()) # Extracting the 'Blood Pressure' column saving to y 

# Independent variables (features)
X = dataset.drop(["Price"], axis=1) # Extracts all other columns except 'Blood Pressure' column

# Check to ensure correct indpendant values are kept
# print(X.head())

# .shape in NumPy tells how many rows and columns to the data held in x and y  
# print("The shape of the independent features is:", X.shape)
# print("The shape of the dependent feature is:", y.shape)

# Print the entire X (independent variables)
# print(X)

# Print the entire y (dependent variable)
# print(y)

# Now create the feature matrix and target vector
X = dataset[['SqFt', 'Bedrooms']].values
y = dataset['Price'].values

# .reshape converts the depentant varible from a row list to a colmn
y = y.reshape(len(y), 1)
print("The shape of the dependent feature is:", y.shape)

# Feature scaling (standardization) for all features in X - (value−mean)/standarddeviation
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Optional standardization of the Y
y = (y - y.mean()) / y.std()

# This adds a column of 1s to X to represent the bias term, the value the model predicts if all inputs are zero.
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

# Checks to ensure data is formatted and ready for training.
# print("X after adding bias term:\n", X)
# print("y after scaling and reshaping:\n", y)

# Using NumPy dataframe to lable the independant varibles, this is optional but benifits readability  
Independent_Variables = pd.DataFrame(X)
Independent_Variables = pd.DataFrame(X, columns=["Age (scaled)", "Weight (scaled)", "Bias"])

# This is a custom version of train_test_split(), similar to the one in sklearn.
# The function "split_data" splits the given dataset into a training set and a test set.

# Parameters:
# X: The independent features (e.g., age, weight, bias)
# y: The dependent variable (e.g., blood pressure)
# test_size: The fraction of data to include in the test set (default is 20%)
# random_state: A seed for reproducibility — ensures the same random split every time the program is run

def split_data(X, y, test_size=0.2, random_state=0):
    np.random.seed(random_state)                  # Set the seed for reproducible results
    indices = np.random.permutation(len(X))       # Shuffle indices to avoid patterns or collection bias
    data_test_size = int(X.shape[0] * test_size)  # Calculate the number of test samples

    # Split the indices into training and testing sets
    test_indices = indices[:data_test_size]       # First part becomes test set
    train_indices = indices[data_test_size:]      # Remaining part becomes training set

    # Assign X and y to their respective splits
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test

class multipleLinearRegression():

  def __init__(self):
    pass

  # Forward pass and compute the prediction and loss
  def forward(self, X, y, W):
    # Parameters:
    # X (array) : Independent Features
    # y (array) : Dependent Features/ Target Variable
    # W (array) : Weights 

    # Returns:
    # loss (float) : Calculated Squared Error Loss for y and y_pred
    # y_pred (float) : Predicted Target Variable

    # Multiply the weights by the input features to get a prediction
    y_pred = np.dot(W, X)  # predicted blood pressure based on current weights
    # Work out the error by comparing prediction to actual result (Mean Squared Error, dividing by 2 for simpler maths later)
    loss = ((y_pred - y) ** 2) / 2
    # Return the error (loss) and the prediction 
    return loss, y_pred

  # Update model weights using gradient descent
  def updateWeights(self, X, y_pred, y_true, W, alpha):
    # Parameters:
    # X (array) : Independent Varibles 
    # y_pred (float) : Predicted Target Variable (scalar)
    # y_true (float) : Actual Target Variable (scalar)
    # W (array) : Current weights
    # alpha (float) : Learning rate
    # index (int) : Index to fetch the corresponding values of W, X and y

    # Returns:
    # W (array) : Updated Weights

    for i in range(X.shape[0]):  # Loop over each features
        # Calculate the error, and update the wieght to reduce the loss  
        W[i] -= alpha * (y_pred.item() - y_true.item()) * X[i]  # Use .item() to ensure scalar values
    # Returns updated wieghts
    return W

  def train(self, X, y, epochs=10, alpha=0.001):
    # Intilise the number of rows (data entries) and colmns (features) for independant varibles
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    W = np.random.randn(num_cols) / np.sqrt(num_rows)  # This initializes wieghts with small random values, scaled using the 1/SqRt

    # Lists to hold loss and the number of epochs during training
    train_loss = []
    num_epochs = []

    # Indices for shuffling data during each epoch
    train_indices = [i for i in range(X.shape[0])]

    for j in range(epochs):
      cost = 0 # Will keep track of the epoch loss

      # Shuffle the values each epochs to reduce overfitting
      np.random.shuffle(train_indices)
      
      for i in train_indices:
        loss, y_pred = self.forward(X[i], y[i], W)  # Pass to foward() for predictions and loss
        cost += loss # Accumilate loss for the epoch
        W = self.updateWeights(X[i], y_pred, y[i], W, alpha)  # Update weights correctly using gradient decent updateWeights()

      # Tracking loss and epoch number
      train_loss.append(cost) 
      num_epochs.append(j)

      # print(f"Epoch {j+1}/{epochs}, Loss: {cost.item()}")  # Print loss at each epoch

      # Print every 25 epochs
      if (j + 1) % 25 == 0 or (j + 1) == epochs:
          print(f"Epoch {j+1}/{epochs}, Loss: {cost.item()}")
    
    return W, train_loss, num_epochs  # Return weights, loss, and epochs

  def test(self, X_test, y_test, W_trained):
    # List to store predictions and corresponding loss 
    test_pred = []
    test_loss = []
    test_indices = [i for i in range(X_test.shape[0])]
    
    for i in test_indices:
      loss, y_test_pred = self.forward(X_test[i], y_test[i], W_trained) # Pass to foward() for predictions and loss
      # Store scalar values of predictrion and loss
      test_pred.append(y_test_pred.item())
      test_loss.append(loss.item())

    return test_pred, test_loss # Return prediction and loss

  # Prediction function
  def predict(self, W_trained, X_sample): # W_trained- wieghts from training and X_sample new input sample 
    prediction = np.dot(W_trained, X_sample)  # Use dot product to make prediction, mutiplication of two vectors
    return prediction

  # Diisplay the plot loss during training as a graph
  def plotLoss(self, loss, epochs):
    plt.plot(epochs, loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Plot Loss')
    plt.show()

# Splitting the dataset into training and testing sets
X_train, y_train, X_test, y_test = split_data(X, y)

# Print the shape of the datasets for training and testing
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Declaring the "regressor" as an object of the class multipleLinearRegression
regressor = multipleLinearRegression()

# Train the model
W_trained, train_loss, num_epochs = regressor.train(X_train, y_train, epochs=200, alpha=0.0001)

# Testing on the Test Dataset
test_pred, test_loss = regressor.test(X_test, y_test, W_trained)

# Print Test Predictions and Losses (optional)
print("Test Predictions:", test_pred)
print("Test Losses:", test_loss)

# Plot the Train Loss
regressor.plotLoss(train_loss, range(1, 201))  # Use range for epochs to plot correctly

# Define real min and max values for each feature from the dataset
sqft_min, sqft_max = 1590, 2590       # Square Foot range
bed_min, bed_max = 2, 4               # Bedrooms range
price_min, price_max = 124000, 199500  # Price range

def inverse_scale_sqft(scaled):
    return scaled * sqft_std + sqft_mean

def inverse_scale_bed(scaled):
    return scaled * bed_std + bed_mean

def inverse_scale_price(scaled):
    return scaled * price_std + price_mean


# 3D Plotting with proper labels and scaling
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot actual training data
actual_sqft = inverse_scale_sqft(X_train[:, 0])
actual_beds = inverse_scale_bed(X_train[:, 1])
actual_price = inverse_scale_price(y_train.flatten())


ax.scatter(actual_sqft, actual_beds, actual_price, color='r', label='Actual')

# Predicted prices
predicted_train_y = np.dot(X_train, W_trained)
predicted_price = inverse_scale_price(predicted_train_y.flatten())

ax.scatter(actual_sqft, actual_beds, predicted_price, color='g', label='Predicted')

# Plot regression surface
x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10),
                             np.linspace(X[:, 1].min(), X[:, 1].max(), 10))

z_surf_input = np.c_[x_surf.ravel(), y_surf.ravel(), np.ones_like(x_surf.ravel())]
z_surf = z_surf_input.dot(W_trained).reshape(x_surf.shape)

# Rescale the surface to the original blood pressure range
predicted_surf_hp = inverse_scale_price(z_surf)

# Plot the regression surface
ax.plot_surface(inverse_scale_sqft(x_surf), inverse_scale_bed(y_surf), predicted_surf_hp, alpha=0.5, color='blue')

# Labels and title
ax.set_xlabel('Square Foot')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('House Price')
ax.set_title('3D Result Graph Multiple Linear Regression')

# Show the legend and plot
ax.legend()
plt.show()

