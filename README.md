# neuraltree

This is an attempt to build a differentiable decision tree with neural networks as branches and leaves. Each non-leaf node in a tree has a network and some children. The network produces a list of probabilities, one for each child. The leaves of the tree have networks too, but these networks output final classifications. The goal when training the tree is to maximize the probability of outputting the correct classification.

# TODO

 * Use log of softmax instead of softmax itself, allowing numerically stable cross entropy.
