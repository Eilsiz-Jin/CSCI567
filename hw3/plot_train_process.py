import matplotlib.pyplot as plt
import json
import numpy as np

data = json.load(open("MLP_lr0.01_b5.json"))
plt.title("Training accuracy vs. epoch on a batch size of 5")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(np.arange(len(data['train'])), data['train'])
plt.savefig("train_acc_cmd.png")
plt.show()