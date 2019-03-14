# Adaptative concentration of Regression Tree

This code is the implementation of the Guess And Check procedure as described in the article of Wager et al. in https://arxiv.org/abs/1503.06388.

This procedure is supposed to give theoretical bounds on the prediction of the tree. Even though the result holds at infinity, we investigate its use for large samples. 

## To do

* Check validity of rectangle representation
* Provide Bayes Classifier representation when there's a single useful variable.

## Features 

### Using the model 

* Create it with `GuessAndCheck()`
* Fit it with the `fit` method.
* Predict it accordingly

### Visualization 

* Create the graph of the tree with the `show_graph()` method. Needs Networkx.
* Create the representation of the bounding rectangles for the first two dimensions with 
`plot_bounding_boxes()` - *not sure if that works*.
* The `used_vars` attribute contains a dictionary which counted the occurence of every variables used. 
Useful to see if the variables on which Y depends where indeed the most selected or not, with a 

## Author's citation

*Adaptive Concentration of Regression Trees, with Application to Random Forests*
**Wager, Stefan; Walther, Guenther**
We study the convergence of the predictive surface of regression trees and forests. To support our analysis we introduce a notion of adaptive concentration for regression trees. This approach breaks tree training into a model selection phase in which we pick the tree splits, followed by a model fitting phase where we find the best regression model consistent with these splits. We then show that the fitted regression tree concentrates around the optimal predictor with the same splits: as d and n get large, the discrepancy is with high probability bounded on the order of sqrt(log(d) log(n)/k) uniformly over the whole regression surface, where d is the dimension of the feature space, n is the number of training examples, and k is the minimum leaf size for each tree. We also provide rate-matching lower bounds for this adaptive concentration statement. From a practical perspective, our result enables us to prove consistency results for adaptively grown forests in high dimensions, and to carry out valid post-selection inference in the sense of Berk et al. [2013] for subgroups defined by tree leaves.

