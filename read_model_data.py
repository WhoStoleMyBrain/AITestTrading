import json
import matplotlib.pyplot as plt
from glob import glob

def load_and_visualize(filename):
    with open(filename, 'r') as file:
        model_data = json.load(file)

    epochs = list(range(1, model_data["Epochs"] + 1))
    plt.figure(figsize=(14, 4))
    plt.plot(epochs, model_data["Train Loss Over Epochs"], label="Train Loss", color="blue")
    plt.plot(epochs, model_data["Test Loss Over Epochs"], label="Test Loss", color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Test Loss Over Epochs\nModel: {model_data["Model Structure"]}\nFeatures: {", ".join(model_data["Features Used"])}')
    plt.legend()
    plt.grid(True)
    plt.show()

all_config_files = glob("results/*.json")
all_config_files.sort()
for file in all_config_files[::-1]:
    load_and_visualize(file)
    break
# Example usage:
# load_and_visualize("results/model_data_YYYYMMDD_HHMM.json")
