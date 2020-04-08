# Active Learning
Active learning is a form of semi-supervised machine learning where the algorithm can choose which data it wants to learn from.
You can think Active Learning is a kind of design methodology similar to transfer learning.
Which uses small amount of labelled data for learning purpose. It will give you a very powerful tool which can be used 
when there is a shortage of labelled data.

Active learning query method was able to select such good points is one of the major research areas within active learning.
This learning algorithm can choose the data it wants to learn from.

Active learning, there are typically three scenarios or settings in which the learner will query the labels of instances.
1. Membership Query Synthesis
2. Stream-Based Selective Sampling
3. Pool-Based sampling


We are going to consider 4 different strategies for building these subsets of data from the original training set
1. Random sampling
2. Uncertainty sampling
3. Entropy sampling
4. Margin sampling
