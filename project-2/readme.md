Patrick Davlin
Project 2

This file can be viewed in rich format on Github:

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

>It's difficult to say for sure, but generally speaking there are a few improvements that could be made to the results of this experiment:
> 
> 1. Changing the training/validation split in the incoming dataset
> 2.  
> 3.

```
=====DEFAULT=====
Testing classifier results...
Default settings
accuracy: 0.41332975006718625. 6 false positives, 0 false negatives
max_depth = 5
accuracy: 0.3388873958613276. 4 false positives, 0 false negatives
max_features
accuracy: 0.4157484547164741. 3 false positives, 1 false negatives
=====GINI INDEX=====
top fifty features: [ 74   5   7  43  52  50  57  64  93  99  66  80  31 114 121 108  61  92
  46  49  82  17 115  38 104 107  37 105  41  76  36 120  98  25  23 116
  69  65  71  77  20  78  97 118  48  15  96  73  67  90]
Testing classifier results...
Default settings
accuracy: 0.13625369524321418. 10 false positives, 14 false negatives
max_depth = 5
accuracy: 0.12469766191883902. 3 false positives, 1 false negatives
max_features
accuracy: 0.13786616500940607. 7 false positives, 12 false negatives
=====CONDITIONAL ENTROPY=====
top fifty features: [  6 122  53  27  14  79  84  60  32  70  45  21  59  12   2  30   0 103
  88  85  68 117 112  83  81  24  11  26  33  75   9 102  29  54 111  87
 101  91  55 109  19   1  63  95  39   4  47  44  16  18]
Testing classifier results...
Default settings
accuracy: 0.37248051599032517. 5 false positives, 0 false negatives
max_depth = 5
accuracy: 0.3259876377317925. 3 false positives, 0 false negatives
max_features
accuracy: 0.3705993012631013. 4 false positives, 0 false negatives
=====POINTWISE MUTUAL INFORMATION=====
top fifty features: [ 22  59  53 107  68  62  42 100 121  88  72 122  40 110 106  11 114 119
  70  29  26  86  81 103  85  84 113  79 117  33  55 112  19  47  87 101
  91  60  39 102  16  18  44  75 109  95  54  63  83 111]
Testing classifier results...
Default settings
accuracy: 0.3554152109647944. 1 false positives, 5 false negatives
max_depth = 5
accuracy: 0.3218220908357968. 3 false positives, 0 false negatives
max_features
accuracy: 0.34990593926363883. 2 false positives, 3 false negatives
```