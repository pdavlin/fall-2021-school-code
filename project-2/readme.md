Patrick Davlin
Project 2

This file can be viewed in rich format on [Github](https://github.com/pdavlin/fall-2021-school-code/tree/main/project-2)

[Results at bottom of file]

**Part 1 Notes**

There are several parameters for the RandomForestClassifier. In the process of working on this project, a few of them were assessed. In the final project submission, the three parameter settings were as follows:

1. default: chosen to get a baseline performance
2. `max depth = 5`: ordinarily, the classifier will expand nodes of the tree until all the nodes have been expanded or marked as pure. Choosing to limit to a max depth of any value (in this case, 5) allows for faster execution at the cost of some accuracy (generally in the range of 7%-9%).
3. `max_samples=0.5, max_leaf_nodes=5`: I wanted to see whether reducing the max samples by 50% and imposing a limit on the maximum leaf nodes made a meaningful impact on accuracy. Like the max depth calculation, this proved to reduce accuracy fairly significantly across the board.

Tests were run with the same seed in all cases so that results could be compared across the different parts of the experiment. Additionally, the classifier was effectively run in parallel as much as possible to make use of ample computing capacity.

**Analysis**

1. *What do you think about the performance of different executions of the RandomForest classifier? Which execution gave the best results and which one gave the worst results?*

> The executions of the RandomForestClassifier did not seem to differ meaningfully across the board, except for the Gini method. I think that in general this is the case because the classifier is already implementing some form of gini or entropy checking in its runtime, and this assignment was trying to implement those functions before the classifier was run. All the methods also seemed to *decrease* the accuracy of the classifier, due to the fact that eliminating terms being considered by the classifier served to reduce the number of connections between the vectors. The worst results by far were the Gini results; this is most likely due to the fact that the algorithm there was the most difficult to implement properly but also because it's the same method that the RandomForestClassifier uses by default for purity calculations.

2. Provide three ways in which you can improve the results from your experiment.

>It's difficult to say for sure without trying them thoroughly, but generally speaking there are a few improvements that could be made to enhance the results of this experiment:
> 
> 1. Changing the training/validation split in the incoming dataset. Past assignments have indicated this has a significant impact on the effectiveness of classifiers.
> 2. Testing different subset sizes for the result of the feature selection algorithms. Reducing to 50 terms seems like it made the classifier less effective in general; it would be interesting to determine whether there's a tradeoff point between minimizing the amount of terms input to the classifier and maximizing accuracy.
> 3. Combining and comparing feature selection methods. Utilize more than one feature selection methods and isolate the terms that represent a union of their results; columns that appear in two or three feature selections may be more effective in improving the classifier.


**Results**
```
=====DEFAULT=====
Testing classifier results (seed 99)...
Default settings
accuracy: 0.41655468959957. 2 false positives, 0 false negatives
max_depth = 5
accuracy: 0.3417092179521634. 5 false positives, 0 false negatives
50% max_samples
accuracy: 0.32827196990056434. 4 false positives, 0 false negatives
=====GINI INDEX=====
Testing classifier results (seed 99)...
Default settings
accuracy: 0.13625369524321418. 8 false positives, 8 false negatives
max_depth = 5
accuracy: 0.1263101316850309. 4 false positives, 0 false negatives
50% max_samples
accuracy: 0.11421660843859177. 5 false positives, 0 false negatives
=====CONDITIONAL ENTROPY=====
Testing classifier results (seed 99)...
Default settings
accuracy: 0.3723461435098092. 5 false positives, 1 false negatives
max_depth = 5
accuracy: 0.32585326525127656. 4 false positives, 0 false negatives
50% max_samples
accuracy: 0.3137597420048374. 4 false positives, 0 false negatives
=====POINTWISE MUTUAL INFORMATION=====
Testing classifier results (seed 99)...
Default settings
accuracy: 0.3500403117441548. 1 false positives, 4 false negatives
max_depth = 5
accuracy: 0.32262832571889277. 3 false positives, 0 false negatives
50% max_samples
accuracy: 0.31013168503090566. 4 false positives, 0 false negatives