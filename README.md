# Adaptative concentration of Regression Tree

This code is the implementation of the Guess And Check procedure as described in the article of Wager et al. in https://arxiv.org/abs/1503.06388.

This procedure is supposed to give theoretical bounds on the prediction of the tree. Even though the result holds at infinity,
we investigate its use for large samples. 

## To do

* Give bounds according to the theorem 
* Provide Bayes Classifier representation when there's a single useful variable.
* Turn the trees into forests

## Features 

### Using the model 

* Create it with `GuessAndCheck()`
* Fit it with the `fit` method.
* Predict it accordingly

### Visualization 

* Create the graph of the tree with the `show_graph()` method. Needs Networkx.
* Create the representation of the bounding rectangles for the first two dimensions with 
`plot_bounding_boxes()`.
* The `used_vars` attribute contains a dictionary which counted the occurence of every variables used. 
Useful to see if the variables on which Y depends where indeed the most selected or not, with a 
matplotlib barplot for instance. 
