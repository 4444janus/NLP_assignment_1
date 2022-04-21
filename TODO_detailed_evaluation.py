# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd

with open("experiments/base_model/model_output.tsv") as file:
    data = pd.read_csv(file, sep='\t', header=None)


print(data.iloc[:,2])