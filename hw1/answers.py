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
In the case of ${\Delta} < 0$, the loss will simply be whether the results undershot the margin, and the gradient will still point towards higher difference from the prediction to the label. Thus when running SGD even when ${\Delta} < 0$, the model will still lower the loss.

"""

part2_q2 = r"""
**Your answer:**
The weights matrix suggest that the model is learning to classify horizonal lines in different parts of the image, and when it finds lines that are indicative of a certain digit, it will strongly push towards that digit.

This is could explain why it sometimes mistakes 5 vs 2 (since there are similar horizontally but flipped), and 4 vs 9.

"""

part2_q3 = r"""
**Your answer:**

The learning rate chosen looks good and not too big or too small.
If it were too small, the loss would barely go down and would possibly converge to a local minimum and stay there after many epochs, rather than slowly approaching 0.

If it were too big, the loss would jump up and down with large "spikes" of learning and unlearning because it jumps from optimum points too quickly and doesn't learn in the long term.

This graph appears to converge steadily towards zero after honing into a local minimum early, and this is why the learning rate appears Good.

1.Slightly underfitted to the training set

Because we see that after the initial spike of learning, the performance on the training set is stagnating while the performance on the validation set is increasing. This means overall good generalization but underfitting the details of the training set and thus not able to improve further.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
