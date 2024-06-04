#!/usr/bin/env python3
'''
Created on May 11, 2024

Author: jrade
'''

import sys  # import the sys module for command-line arguments
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

class Dataset:
    @staticmethod
    def load_dataset(db_file):
        conn = sqlite3.connect(db_file)  # Establish a database connection
        cur = conn.cursor()  # Create a cursor object to interact with the database

        # Load the Details table
        cur.execute('SELECT * FROM Details')  # Execute a SQL query to select all rows from the Details table
        details_rows = cur.fetchall()  # Fetch all the rows
        details_df = pd.DataFrame(details_rows)  # Convert the rows to a pandas DataFrame
        details_df['id'] = range(1, len(details_df) + 1)  # Add a temporary 'id' column with incrementing values starting from 1

        # Load the Bible table
        cur.execute('SELECT * FROM Bible')  # Execute a SQL query to select all rows from the Bible table
        bible_rows = cur.fetchall()  # Fetch all the rows
        bible_df = pd.DataFrame(bible_rows)  # Convert the rows to a pandas DataFrame
        bible_df['id'] = range(len(details_df) + 1, len(details_df) + len(bible_df) + 1)  # Add a temporary 'id' column with incrementing values starting from the last 'id' of the Details dataframe

        conn.close()  # Close the database connection
        return details_df, bible_df  # Return the two DataFrames

class Preprocessor:
    @staticmethod
    def preprocess_data(details_df, bible_df):
        # Implement any preprocessing steps needed
        # For example, normalize or transform data
        # ...
        return details_df, bible_df

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
class Modeel:
    pass

class ModelHandler:
    @staticmethod
    def load_model(model_path):
        """Load the model state dict from SafeTensors file"""
        state_dict = load_file(model_path)
        if not model_path:
            model = SimpleModel()
        ele # Ensure this matches the saved model architecture
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
        return model

    @staticmethod
    def save_model(model, model_path):
        """Save the model state dict to SafeTensors file"""
        state_dict = model.state_dict()
        save_file(state_dict, model_path)

class Trainer:
    @staticmethod
    def train_model(train_data):
        # Instantiate the model, loss function, and optimizer
        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Prepare the DataLoader
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Training loop
        losses = []
        for epoch in range(100):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss)
                loss.backward()
                optimizer.step()
            #print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        print('Training complete')
        avg_loss = sum(losses) / len(losses)
        std_dev = (sum((x - avg_loss) ** 2 for x in losses) / len(losses)) ** 0.5
        high = max(losses)
        low = min(losses)
        print(f"Average Loss: {avg_loss}")
        print(f"Standard Deviation of Loss: {std_dev}")
        print(f"High: {high}")
        print(f"Low: {low}")
        return model    

class GPUHandler:
    @staticmethod
    def move_model_to_device(model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            count = torch.cuda.device_count()
            if count > 1:
                print('Moving model to cuda:n+1')
                model = torch.nn.DataParallel(model)
            else:
                print('Moving model to cuda:0')
                model = model.to(device)
        return model
    
def search_file(directory, file_name):
    for root, dirs, files in os.walk(directory):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError(f"{file_name} not found in directory {directory}")

def main():
    '''
    if len(sys.argv) != 3:
        print("Usage: python script.py <db_file> <model>")
        sys.exit(1)
    db_file = sys.argv[1]
    model_file = sys.argv[2]
    db_path = os.path.normpath(os.path.join('../../resources/datasets/mybible/bibles/', db_file))
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    print(f"Database file found: {db_path}")
    model_path = os.path.normpath(os.path.join('../../resources/datasets/model/', model_file))
    
    if os.path.exists(model_path):
        model_path = os.path.normpath('../../resources/models/SimpleModel/model.safetensors')  # Specify the file path for the model
    '''

if len(sys.argv) != 3:
    print("Usage: python main.py <db_file> <resource_file>")
    sys.exit(1)
resources_path = os.path.normpath(r'../../../../resources/') 
db_file = sys.argv[1]
model_file = sys.argv[2]  
db_path = search_file(resources_path, db_file)
model_path = search_file(resources_path, model_file)
    
print('Loading Model')

model = ModelHandler.load_model(model_path)
model = Model()

print('Loading Dataset')
details_df, bible_df = Dataset.load_dataset(db_path)

print('Running Preprocessor')
details_df, bible_df = Preprocessor.preprocess_data(details_df, bible_df)

    # Combine the dataframes into one TensorDataset for the SimpleModel training example
    # Assuming you have numeric data that can be converted to tensors
    # Example dummy data:
train_data = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))

print('Training Model')
model = Trainer.train_model(train_data)  # Train the model

print('Saving Updated Model')
ModelHandler.save_model(model, model_path)

print('Model is ready and updated')
exit(0)

if __name__ == '__main__':
    main()
