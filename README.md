# Pipeline for ML or DS Projcts

- Unit Testing > Break your code into small segments, and verify that each segment works. [Article](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765)
- Bug check > Make sure the code is free of bug(s) before tuning the networks for performance.
- Use the appropriate loss
- Measure loss on the correct scale
- Never scale the test data using the statistis of the test data.
- Build a small network and verify that it works. Incrementally increase model complexity, and verify each adddition works.
- Haveing a few more neurons makes it easier for the optimizer to find a good configuration > Too few neurons in a layer can restrict the representation that the network learns.
- The number of hidden layers > Too many hidden layers can risk overfitting and make it very hard to optimize the network
- Choosing a clever network wiring can do a lot of work for you.
- Pay attention to network weights/biases initialization. Initialization over too-large an interval can set initial weights too large, meaning that single neurons have an outsize influence over the network behavior.
- Choose the proper activation funtions for each layer.
- The objective function of a neural network is only convex when there are no hidden units 
- What is the optimal value of learning rate?
- Learning rate scheduling
- What is the optimal value of batch size > Large batch size has smaller variance. We want the batch size to be large enough to be informative about the direction of the gradient, but small enough that the optimizer can regularize the network.
- The scale of the data can make a big difference on training > Standardizing the data to have 0 mean and unit variance, or to lie in a small interval like [-0.5, 0.5] can improve training.
- Batch or Layer normalization can improve network training > Both seek to improve the network by keeping a running mean and standard deviation for neurons activations as the network trains. 
- Only employ regularization when your network is able to learn and learn well. Add regularization to improve the network performance on validation set without hurting the performance on the training set > The choice and the parameters of the regularizer(s) now become important. Add regularizers one after another and verify each addition.
- Batch normalization and drop out are difficult to use together. Large values for L1 and L2 regularizer make the weights stagnant.
- Track everything related to experiments and execution. 
- Pay attention to the initial loss > A bad network initialization can lead skewed values of loss. 
- Visualize or analyze the output of each layer.
- Don't use NN for the starting point > Use classical learning algrotihms which are interpretable and understood for initial analysis of learnability. 
- If possible try to use standard NN architecture or networks.
- Reduce the training set to 1 or 2 samples, and train on this. The NN should immediately overfit the training set, reaching an accuracy of 100% on the training set very quickly, while the accuracy on the validation/test set will go to 0%. If this doesn't happen, there's a bug in your code.
- Keep the full training set, but you shuffle the labels. The only way the NN can learn now is by memorising the training set, which means that the training loss will decrease very slowly, while the test loss will increase very quickly. In particular, you should reach the random chance loss on the test set. This means that if you have 1000 classes, you should reach an accuracy of 0.1%. If you don't see any difference between the training loss before and after shuffling labels, this means that your code is buggy (remember that we have already checked the labels of the training set in the step before).
- Visualize or plot every possible thing.
- Thoroughly inspect the data > Visualize examples, understand the distribution, look for patterns, data imbalances and biases, vaiance in the data and ways to mitigate it, detect outliers, clean
- Set up a full training + evaluation skeleton and gain trust in its correctness via a series of experiments
- fix random seeds
- simplify everything
- initialize the final layer weights correctly > E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
- Monitor metrics other than loss that are human interpretable and checkable (e.g. accuracy). Whenever possible evaluate your own (human) accuracy and compare to it. Alternatively, annotate the test data twice and for each example treat one annotation as prediction and the second as ground truth.
- Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.
- Verify the input of the netowrk just before passing the input 
- Get a model large enough that it can overfit (i.e. focus on training loss) and then regularize it appropriately (give up some training loss to improve the validation loss).
- When tuning start with learning rate -> mini_batch_size -> momentum -> #hidden_units -> # learning_rate_decay -> #layers
- If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you’d expect.
- First, the by far best and preferred way to regularize a model in any practical setting is to add more real training data.
- The next best thing to real data is half-fake data - try out more aggressive data augmentation.
- It rarely ever hurts to use a pretrained network if you can, even if you have enough data.
- Remove features that may contain spurious signal. Any added spurious input is just another opportunity to overfit if your dataset is small.
- Due to the normalization inside batch norm smaller batch sizes somewhat correspond to stronger regularization.
- Stop training based on your measured validation loss to catch your model just as it’s about to overfit.
- For simultaneously tuning multiple hyperparameters it may sound tempting to use grid search to ensure coverage of all settings, but keep in mind that it is best to use random search instead.
- Model ensembles are a pretty much guaranteed way to gain 2% of accuracy on anything
- Leave it training even after validation loss plateau




## Links
* [StackOverFlow](https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn)
* [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)





