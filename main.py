import numpy as np
from guess_and_check import GuessAndCheck
import matplotlib.pyplot as plt

n_samples = 5000000
n_total_features = 100
n_good_features = 2

X = np.random.random(size=(n_samples, n_total_features))

# Y = np.linalg.norm(X[:, 0:n_good_features] - (np.zeros((n_samples, n_good_features))+0.5), axis=1)
# Y = np.exp(-Y**2/2)
# max_y = np.exp(-np.power(np.linalg.norm(np.zeros(n_total_features) + 0.5), 2)/2)

Y = X[:, 0] + X[:, 1]
max_y = 2

model = GuessAndCheck(leaf_size=500000, balance_param=0.5, max_y=max_y)
model.fit(X, Y)

if model.n_of_nodes == 1:
    print("Trust border : %f" % model.trust_border)
    print("Failed split : %s" % model.root.failed_split)
else:
    used_var = model.variables_used
    plt.bar(used_var.keys(), used_var.values())
    model.show_graph()

t = model.compute_bounding_boxes()
model.plot_bounding_boxes(t)
