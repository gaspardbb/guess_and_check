import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings
import time


class Node:

    def __init__(self, samples_indices):
        self.samples_indices = samples_indices
        self.size = samples_indices.shape[0]
        self.failed_split = []  # For debug purpose
        self.failed_variables = []

        self.tried = False
        self.is_leaf = False
        self.split_index = None
        self.split_value = None
        self.split_energy = None
        self.left_child = None
        self.right_child = None

        # To draw the graph, we need unique id
        self.id = np.random.randint(np.int(1e6))


class GuessAndCheck:
    """Implementation of the Guess-and-Check procedure. Complexity of fit seems to be : O(n_samples^2 ...)"""

    def __init__(self, leaf_size: int, balance_param: float, max_y):
        if leaf_size <= 0 or not isinstance(leaf_size, int):
            raise ValueError("Leaf size should be a positive integer. Found : %f" % leaf_size)
        elif not 0 <= balance_param <= .5:
            raise ValueError("Balance parameter should be in [0,.5]. Found %f" % balance_param)
        elif max_y <= 0:
            raise ValueError("y should be in [-M;M] with M>0. Found %f" % max_y)

        self.k = leaf_size
        self.alpha = balance_param
        self.m = max_y
        self.successful_variables = []

        self.n_of_nodes = 0
        self.is_fit = False
        self.n_samples = 0
        self.n_features = 0
        self.trust_border = 0
        self.root = None
        self.x = None
        self.y = None
        self.to_be_processed = []

        # Debug
        self.variables_used = dict()
        self.first_run = True

        # To show it
        self.graph = None

    def fit(self, x, y):
        if self.is_fit:
            raise NotImplementedError("Trees should be fitted only once.")

        time_fit_start = time.time()
        self.n_samples, self.n_features = x.shape
        self.trust_border = (2 * 9 * self.m * np.sqrt(
            np.log(self.n_samples) * np.log(self.n_features) / self.k / np.log(1 / (1 - self.alpha))
        )) ** 2
        self.y = y
        self.x = x

        if 2 * self.k > self.n_samples:
            raise ValueError("Minimum leaf size should be lower than half of the training samples. Found : (k, "
                             "n_samples) = (%i, %i)" % (self.k, self.n_samples))
        elif self.n_samples != y.shape[0]:
            raise ValueError("X and y do not have the same shape. Found %i, %i" % (self.n_samples, y.shape[0]))
        if self.k < np.log(self.n_samples) * np.log(self.n_features):
            warnings.warn("Assumption 2 requires that k grows faster than log(n)*log(d). Found : %4.1f, %4.1f"
                          % (self.k, np.log(self.n_samples) * np.log(self.n_features)))

        self.root = Node(np.arange(self.n_samples))
        self.n_of_nodes += 1
        self.to_be_processed = [self.root]

        while self.to_be_processed:
            cur_node = self.to_be_processed.pop()
            # Debug
            assert cur_node.size >= 2 * self.k, "Too few samples in node."
            cur_node.tried = True
            possible_split_for_this_node = [i for i in range(self.n_features) if i not in cur_node.failed_variables]
            split_index = possible_split_for_this_node[
                np.random.randint(len(possible_split_for_this_node))
            ]
            self.find_splitting_point(cur_node, x, y, split_index)

        self.is_fit = True
        if self.n_of_nodes == 1:
            self.root.is_leaf = True
            warnings.warn(
                "Your tree has only one node, which is the root node. You'll have low variance but high bias.")

        all_split = np.int(np.sum([self.variables_used[k] for k in self.variables_used.keys()]))
        correct_split = np.int(np.sum([self.variables_used[k] for k in self.variables_used.keys() if k == 0 or k == 1]))
        print("Tree fitted in %4.1fs!\n"
              "There are %i nodes\n"
              "On the %i split made, %i were on variables 0 and 1."
              % (time.time() - time_fit_start, self.n_of_nodes, all_split, correct_split))

    def find_splitting_point(self, node: Node, x: np.ndarray, y: np.ndarray, split_index: int):
        # First we sort the samples increasingly, so as to check easily the alpha and k rules
        cur_x = x[node.samples_indices, split_index]
        cur_y = y[node.samples_indices]
        cur_x_sorted_indices = np.argsort(cur_x)
        cur_n_samples = cur_x.shape[0]
        # The criterium for complying with the alpha and k rules
        # TODO : check if floor/ceil are in the right order
        start = np.floor(np.max([self.k, self.alpha * cur_n_samples]))
        stop = np.ceil(np.min([cur_n_samples - self.k, (1 - self.alpha) * cur_n_samples]))
        start, stop = np.int(start), np.int(stop)

        if stop - start <= 1:
            # TODO implement the case where there's a single candidate (e.g. alpha = .5)
            pass

        assert stop - start >= 0, "DEBUG : there was a problem computing the possible range of candidates." \
                                  "Found start, stop = %i, %i" % (start, stop)

        best_error = 0
        best_i = -1
        for i in range(start, stop + 1):
            error_theta = compute_squared_error(i, cur_n_samples, cur_y[cur_x_sorted_indices])
            if error_theta > best_error:
                best_error = error_theta
                best_i = i

        if self.first_run:
            warnings.warn("First run terminated. Best error found is %3.1E. Barrier is %3.1E. Ratio is %3.1E."
                          % (best_error, self.trust_border, self.trust_border / best_error), category=RuntimeWarning)
            self.first_run = False

        split_index_is_a_successful_variable = split_index in self.successful_variables
        if split_index_is_a_successful_variable or best_error > self.trust_border:
            if not split_index_is_a_successful_variable:
                self.successful_variables.append(split_index)

            # Debug : we count how many times we use each variable
            if split_index not in self.variables_used.keys():
                self.variables_used[split_index] = 1
            else:
                self.variables_used[split_index] += 1

            node.split_index = split_index
            node.split_value = cur_x[cur_x_sorted_indices[best_i]]
            node.split_energy = best_error

            # Now, we take care of the children.
            # We make sure to provide the good list of the indices of all the elements of the minus group
            left = Node(node.samples_indices[cur_x_sorted_indices[:best_i + 1]])
            if best_i + 1 < 2 * self.k:
                left.is_leaf = True
            else:
                self.to_be_processed.append(left)

            right = Node(node.samples_indices[cur_x_sorted_indices[best_i + 1:]])
            if cur_n_samples - (best_i + 1) < 2 * self.k:
                right.is_leaf = True
            else:
                self.to_be_processed.append(right)

            node.left_child = left
            node.right_child = right

            self.n_of_nodes += 2

        else:
            """If the split fails : 
            - We check whether this split index had already been encountered; if not, we add it to the split_index_which
             _do_not_work_for_this_node list
            - if actually no features can provide a valid split: we say that this node is actually a leaf
            - if that's not the case, we put it again in the to_be_processed list, so that it should finally meet its 
            fate."""
            node.failed_split.append({"split_index": split_index,
                                      "best_index": best_i,
                                      "best_error": best_error})
            if split_index not in node.failed_variables:
                node.failed_variables.append(split_index)
            if len(node.failed_variables) >= self.n_features:
                node.is_leaf = True
            else:
                self.to_be_processed.append(node)

    def predict(self, x):
        if x.shape[1] != self.n_features:
            raise ValueError("Passed array does not have the same nb of features than train array. Found: %i when %i"
                             "was expected." % (x.shape[1], self.n_features))
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            cur_node = self.root
            while not cur_node.is_leaf:
                if x[i, cur_node.split_index] <= cur_node.split_value:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            y_pred[i] = 1 / cur_node.size * np.sum(self.y[cur_node.samples_indices])

        return y_pred

    def make_graph(self):
        if self.n_of_nodes == 1:
            raise ValueError("Can't draw graph on a tree with only one node.")
        graph = nx.DiGraph()
        graph.add_node(self.root.id, split_index=self.root.split_index, split_value=self.root.split_value)
        to_process = [self.root]
        while to_process:
            cur_node = to_process.pop()
            if cur_node.left_child.is_leaf:
                graph.add_node(cur_node.left_child.id, size=cur_node.left_child.size)
                graph.add_edge(cur_node.id, cur_node.left_child.id)
            else:
                graph.add_node(cur_node.left_child.id,
                               split_index=cur_node.left_child.split_index,
                               split_value=cur_node.left_child.split_value)
                graph.add_edge(cur_node.id, cur_node.left_child.id)
                to_process.append(cur_node.left_child)

            if cur_node.right_child.is_leaf:
                graph.add_node(cur_node.right_child.id, size=cur_node.right_child.size)
                graph.add_edge(cur_node.id, cur_node.right_child.id)
            else:
                graph.add_node(cur_node.right_child.id,
                               split_index=cur_node.right_child.split_index,
                               split_value=cur_node.right_child.split_value)
                graph.add_edge(cur_node.id, cur_node.right_child.id)
                to_process.append(cur_node.right_child)

        self.graph = graph

    def show_graph(self):
        if self.graph is None:
            self.make_graph()

        labels = dict(
            (n, "%i, %3.1f" % (d['split_index'], d["split_value"])) if "split_index" in d.keys()
            else (n, str(d["size"]))
            for n, d in self.graph.nodes(data=True))

        nx.drawing.nx_agraph.write_dot(self.graph, "graph.dot")
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, pos, arrows=True)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, arrows=True)

    def compute_bounding_boxes(self):
        # depth first lookup. Use recursive function with dic as parameter.
        def depth_search(node, bounding_box, found_leaves):
            if node.is_leaf:
                found_leaves.append({"leave": node, "bounding_box": bounding_box})
            else:
                if node.left_child:
                    new_box = copy.copy(bounding_box)
                    if node.split_index == 0:
                        new_box["x_max"] = node.split_value
                    elif node.split_index == 1:
                        new_box["y_max"] = node.split_value
                    depth_search(node.left_child, new_box, found_leaves)
                    del new_box  # Not sure if that's necessary
                if node.right_child:
                    new_box = copy.copy(bounding_box)
                    if node.split_index == 0:
                        new_box["x_min"] = node.split_value
                    elif node.split_index == 1:
                        new_box["y_min"] = node.split_value
                    depth_search(node.left_child, new_box, found_leaves)
                    del new_box
                else:
                    raise AssertionError("A node not labeled as a leaf was found to have no children.")

        leaves = []
        box = {"x_min": 0,
               "y_min": 0,
               "x_max": 1,
               "y_max": 1}
        depth_search(self.root, box, leaves)
        return leaves

    def plot_bounding_boxes(self, found_leaves=None):
        if found_leaves is None:
            found_leaves = self.compute_bounding_boxes()
        fig, ax = plt.subplots(1)
        if self.n_samples < 5000:
            sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], ax=ax)
        else:
            warnings.warn("Too many sample points (%1.1E>5000). They were not drawn." % self.n_samples)
        for leaf in found_leaves:
            box = leaf["bounding_box"]
            rect = patches.Rectangle((box["x_min"], box["y_min"]),
                                     box["x_max"] - box["x_min"], box["y_max"] - box["y_min"],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


def compute_squared_error(i, n, y):
    """Compute squared error as explained in the procedure : the l(theta) function.
    :param i: the considered index of the split. Should be in [start, stop], as defined in find_splitting_point to
            comply with the alpha-k splitting rule
    :param n: the number of samples.
            Thus, N^- = i+1, N^+ = n-i+1
    :param y: the **sorted** array of outputs, to match with the corresponding x.
    """
    n_minus = i + 1
    n_plus = n - i + 1
    delta = 1 / n_minus * np.sum(y[:i + 1]) - 1 / n_plus * np.sum(y[i + 1:])
    return 4 * n_minus * n_plus / (n_minus + n_plus) ** 2 * delta ** 2
