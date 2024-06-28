r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. **False**. The in-sample error refers to the training error, which measures how well our model fits
the training set, on which the model was trained on. The test set has nothing to do with the training error.
The test set and training set are disjoint, and the test set is not seen by the model during training. The test
set error (also named the out-of-sample error) measures how well our model "generalizes" i.e. performs on unseen data.

2. **False**. Some train-test splits can be more useful than others. A useful split is a split that creates train and test sets
that represent well the underlying data distribution. But if for example, a split creates a train set with samples of one label, 
it does not represent the overall data well, thus the model trained on it can have poor performance.

3. **True**. Cross-validation should always be conducted on the training set only. In cross-validation, we divide
the training set into 'folds', and use one of the folds as a validation set while the others are used for training the model.
This process is repeated by using each fold as a validation set. The test set is used only after the model training, to 
measure the model's performance on unseen data.

4. **True**. Following the previous explanation of cross-validation, the validation set is not seen by the model when it is 
trained on the other folds, thus the performance on the validation set is used to estimate the models' performance
on unseen data (generalization error). We look at the average performance on the validation sets to better estimate the 
models' generalization error.

"""

part1_q2 = r"""
**Your answer:**

My friend's approach is **not** justified.
When tuning the model hyperparameters, we should only measure the performance on the train set (for example by using 
cross-validation to select the best $\lambda$).
The test set is only used as the final independent evaluation of our model on unseen data (generalization error).
We contaminate the learning process by using the test set to select the best $\lambda$, leading to a biased 
model when it is finally evaluated on the test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
The choice of ${\Delta}$ is arbitrary because if we changed ${\Delta}$ by a factor of $a$, then we could also change the weights by a factor of $a$ and change ${\lambda}$ down by a factor of $a$, which would overall make the loss equivalant.

"""

part2_q2 = r"""
**Your answer:**
The weights matrix suggest that the model is learning the shape of each digit, for example for 0 and 7 you can't clearly see the digit in the weight matrix (noisy weights). There are some digits where the digit position shifts a lot and the weights can't capture that accurately like in 2 and 5.

This is could explain why it sometimes mistakes 5 vs 2, and 4 vs 9.

"""

part2_q3 = r"""
**Your answer:**

The learning rate chosen looks good and not too big or too small.
If it were too small, the loss would barely go down and would possibly converge to a local minimum and stay there after many epochs, rather than slowly approaching 0.

If it were too big, the loss would jump up and down with large "spikes" of learning and unlearning because it jumps from optimum points too quickly and doesn't learn in the long term.

This graph appears to converge steadily towards zero after honing into a local minimum early, and this is why the learning rate appears Good.

1.Slightly underfitted to the training set

Because we see that after the initial spike of learning, the performance on the training set is stagnating while the performance on the validation set is increasing. This means overall good generalization but slight underfitting the details of the training set and thus not able to improve further.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is randomly scattered residuals centered around the zero line,
with approximately constant variance across the predicted values.
There should be no clear pattern of increase or decrease in variance.
Based on the residual plot I got after CV, I can say that the fitness of the trained model is relatively high compared to
the top-5 features model. We can see that the final plot after CV has a better residual
pattern than the top-5 feature's plot - the residuals are closer and more centered around the zero line (as is also 
indicated by the lower MSE error), having similar variance across the predicted values.
Also, there are significantly fewer outliers.

"""

part3_q2 = r"""
**Your answer:**

1. It is still a linear regression model because it is still a linear combination of the features - the model
is still a linear function (where $W$ stores the coefficients). Only the non-linear features were produced 
by applying a non-linear function to the original features.

2. Technically we can apply any non-linear on the original features, but it will benefit the model if it
captures the underlying non-linear relationships. We have to keep in mind that adding features increases the dimension
which can lead to overfitting, and some non-linear functions can have a high computational cost.

3. The decision boundary is still a hyperplane in the **transformed** feature space since the model is
a linear combination of the transformed features. However, the decision boundary becomes non-linear
in the **original** feature space. For example, if we originally had two features $x_1,x_2$ and transformed them
to $x_1^2, x_2$, then the decision boundary is still a hyperplane in the transformed feature space: $x_1^2, x_2$
but in the original feature space $x_1, x_2$ the decision boundary is a parabola.
"""

part3_q3 = r"""
**Your answer:**

1. Using log space allows us to sample $\lambda$ values from a larger range, where the sampling is proportionate to
the scale (ensuring that small and large values are sampled according to their relative scale).
Using linear space is not efficient for sampling values from a large range, as the values are linearly spaced.
It will sample many values that will be too close to each other, making some of the samples redundant. If
we reduce the number of samples by decreasing the range or by increasing the spacing, we can miss important 
$\lambda$ values.

2. The model was fitted to the data 180 times.
Calculation:
- The degrees range has 3 values ([1,2,3]).
- The $\lambda$ range has 20 values.
- The total number of degree + $\lambda$ combinations is: $3 * 20 = 60$. This is the number of times cross validation is 
performed (since at each CV we check one combination).
- At each CV, since the number of folds is 3, we fit the model 3 times (on each combination of 2 folds 
while the third fold is the validation set)
- Thus overall the number of times the model was fit on the data is: $(3 * 20) * 3 = 180$

"""

# ==============

# ==============
