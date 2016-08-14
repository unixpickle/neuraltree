# neuraltree

This is a differentiable decision tree with neural networks as branches and leaves. Trees of this form have more power than single layers of perceptrons, but I have yet to compare them to other deep architectures.

# Algorithm

Each non-leaf node in a tree has a network and some children. The network produces a list of probabilities, one for each child. The leaf nodes have networks too, but these networks output final classifications. The goal when training the tree is to maximize the probability of outputting the correct classification. The tree can be trained easily using classical back propagation and a variant of [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
