import matplotlib.pyplot as plt
import json
import numpy as np

test_acc = []
train_time = []
bs = [1, 5, 50, 500, 5000]
for i in bs:
    data = json.load(open("MLP_lr0.01_b%s.json" % str(i)))
    test_acc.append(data["test"])
    train_time.append(data["time"])
plt.title("Test accuracy vs. batch size")
plt.xlabel("Batch size")
plt.ylabel("Test accuracy")
plt.plot(np.arange(5), test_acc)
plt.xticks(np.arange(5), bs)
plt.savefig("test_acc.png")
plt.show()

plt.title("Training time vs. batch size")
plt.xlabel("Batch size")
plt.ylabel("Training time")
plt.plot(np.arange(5), train_time)
plt.xticks(np.arange(5), bs)
plt.savefig("train_time.png")
plt.show()