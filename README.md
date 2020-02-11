# Machine Learning Interview Questions and Answers

This list was compiled while preparing for AI Residency programs in Google, Facebook, Microsoft, Uber etc.

## Linear Algebra

#### 1. What is broadcasting in connection to Linear Algebra

The term broadcasting describes how numpy treats arrays with 
different shapes during arithmetic operations. Subject to certain 
constraints, the smaller array is “broadcast” across the larger 
array so that they have compatible shapes. Broadcasting provides a 
means of vectorizing array operations so that looping occurs in C
instead of Python. It does this without making needless copies of 
data and usually leads to efficient algorithm implementations.

#### 2. What are scalars, vectors, matrices, and tensors

Scalar: A single number.
Vector : A list of values.(rank 1 tensor)
Matrix: A two dimensional list of values.(rank 2 tensor)
Tensor: A multi dimensional matrix with rank n.

#### 3. What is Hadamard product of two matrices

Hadamard product is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands where each element i, j is the product of elements i, j of the original two matrices.

#### 4. What is an inverse matrix

The inverse of a matrix A is a matrix that, when multiplied by A results in the identity. Invertible matrices have connections back to systems of equations and to other concepts like linear independence or dependence.

#### 5. If inverse of a matrix exists, how to calculate it

Step 1: calculating the Matrix of Minors,
Step 2: then turn that into the Matrix of Cofactors,
Step 3: then the Adjugate, and
Step 4: multiply that by 1/Determinant.

#### 6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?

The determinant is a scalar value that can be computed from the elements of a square matrix and encodes certain properties of the linear transformation described by the matrix. Pick any row or column in the matrix. Multiply every element in that row or column by its cofactor and add. The result is the determinant. If A is an n × n matrix, then the sum of the n eigenvalues of A is the trace of A and the product of the n eigenvalues is the determinant of A.

#### 7. Discuss span and linear dependence

If we use a linearly dependent set to construct a span, then we can always create the same infinite set with a starting set that is one vector smaller in size. However, this will not be possible if we build a span from a linearly independent set. So in a certain sense, using a linearly independent set to formulate a span is the best possible way — there are not any extra vectors being used to build up all the necessary linear combinations.

#### 8. What is Ax = b? When does Ax =b has a unique solution

If and only if b is a linear combination of the columns of A

#### 9. In Ax = b, what happens when A is fat or tall

The homogeneous system Ax = 0 has infinitely many solutions. The statement would be true for fat and singular
square matrices, but fails for a nonsingular square matrix.

#### 10. When does inverse of A exist

Inverse of a matrix exists when the matrix is invertible. Now for a matrix to be invertible , you need to have the condition that the determinant of the matrix must not be zero. That is det(A) ≠ 0 where A is your matrix of interest.

#### 11. What is a norm? What is L1, L2 and L infinity norm

L1 Norm is the sum of the magnitudes of the vectors in a space. It is the most natural way of measure distance between vectors, that is the sum of absolute difference of the components of the vectors. L2 norm gives the shortest distance to go from one point to another. L infinity norm gives the largest magnitude among each element of a vector.

#### 12. What are the conditions a norm has to satisfy

If norm of x is greater than 0 then x is not equal to 0 (Zero Vector) and if norm is equal to 0 then x is a zero vector. 

#### 13. Why is squared of L2 norm preferred in ML than just L2 norm

On raising power to cube (L3), quad (L4) or higher, the function becomes sensitive to the influence of outliers and thus introduces unwanted bias into distance calculation. Using squared L2 norm, the function becomes easily calculable.

#### 14. When L1 norm is preferred over L2 norm

L1 is used for feature selection, dealing with sparsity and has less computational cost.

#### 15. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?

Yes, although it is actually not a norm. It corresponds to the total number of nonzero elements in a vector.

#### 16. What is Frobenius norm?

The Frobenius norm is matrix norm of a matrix defined as the square root of the sum of the absolute squares of its elements.

#### 17. What is a diagonal matrix?

A matrix having non-zero elements only in the diagonal running from the upper left to the lower right.

#### 18. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?

#### 19. At what conditions does the inverse of a diagonal matrix exist?

The inverse of a diagonal matrix is obtained by replacing each element in the diagonal with its reciprocal, as illustrated below for matrix C. where I is the identity matrix. 

#### 20. What is a symmetrix matrix?

A symmetric matrix is a square matrix that is equal to its transpose. 

#### 21. What is a unit vector?

A vector which has a magnitude of one.

#### 22. When are two vectors x and y orthogonal?

Two vectors are orthogonal if and only if their dot product is zero, i.e. they make an angle of 90° (π/2 radians), or one of the vectors is zero.

#### 23. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?

Given several non-zero vectors, if they are orthogonal (one to another), then they are linearly independent. So their number cannot exceed 𝑛 if they are in ℝ𝑛 .

#### 24. When are two vectors x and y orthonormal

To summarize, for a set of vectors to be orthogonal : They should be mutually perpendicular to each other

#### 25. What is an orthogonal matrix? Why is computationally preferred?

An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (i.e., orthonormal vectors), i.e. where. is the identity matrix. Because it is invertible.

#### 26. What is eigendecomposition, eigenvectors and eigenvalues?

Eigendecomposition of a matrix is a type of decomposition that involves decomposing a square matrix into a set of eigenvectors and eigenvalues. One of the most widely used kinds of matrix decomposition is called eigendecomposition, in which we decompose a matrix into a set of eigenvectors and eigenvalues. An eigenvector or characteristic vector of a linear transformation is a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it. The corresponding eigenvalue is the factor by which the eigenvector is scaled.

27. How to find eigen values of a matrix?




## Model and feature selection

#### 1. Why are model selection methods needed?

Model selection refers to the problem of selecting a few representative models from a large set of computational models for the purpose of decision making or optimization under uncertainty.

#### 2. How do you do a trade-off between bias and variance?

Bias is the simplifying assumptions made by the model to make the target function easier to approximate.
Variance is the amount that the estimate of the target function will change given different training data.
Trade-off is tension between the error introduced by the bias and the variance.

#### 3. What are the different attributes that can be selected by model selection methods?

Attribute selection measure is a heuristic for selecting the splitting criterion that “best” separates a given data partition, D, of a class-labeled training tuples into individual classes.

#### 4. Why is cross-validation required?

Cross Validation is a very useful technique for assessing the effectiveness of your model, particularly in cases where you need to mitigate overfitting. It is also of use in determining the hyper parameters of your model, in the sense that which parameters will result in lowest test error.

#### 5. Describe different cross-validation techniques.

Cross Validation is a technique which involves reserving a particular sample of a dataset on which you do not train the model. Later, you test your model on this sample before finalizing it.

#### 6. What is hold-out cross validation? What are its advantages and disadvantages?

Hold-out is when you split up your dataset into a 'train' and 'test' set. The training set is what the model is trained on, and the test set is used to see how well that model performs on unseen data.

#### 7. What is k-fold cross validation? What are its advantages and disadvantages?

K fold cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into.
Advantages:
(1) No randomness of using some observations for training.
(2) As validation set is larger than in LOOCV, it gives less variability in test-error as more observations are used for each iteration's prediction.
Disadvantages:
The disadvantage of this method is that the training algorithm has to be rerun from scratch k times, which means it takes k times as much computation to make an evaluation.

#### 8. What is leave-one-out cross validation? What are its advantages and disadvantages? 

Leave-one-out cross validation is K-fold cross validation taken to its logical extreme, with K equal to N, the number of data points in the set. That means that N separate times, the function approximator is trained on all the data except for one point and a prediction is made for that point.

#### 9. Why is feature selection required?

Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.

#### 10. Describe some feature selection methods.

There are two main types of feature selection techniques: wrapper and filter methods.
Filter-based feature selection methods use statistical measures to score the correlation or dependence between input variables that can be filtered to choose the most relevant features.

#### 11. What is forward feature selection method? What are its advantages and disadvantages?

Forward Selection: Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.

#### 12. What is backward feature selection method? What are its advantages and disadvantages?

Backward selection starts with all features contained in the dataset. It then runs a model and calculates a p-value associated with the t-test or F-test of the model for each feature. The feature with the largest insignificant p-value will then be removed from the model, and the process starts again.

#### 13. What is filter feature selection method and describe two of them?

Filter method: selecting a subset of features by a measure other than error (a measure that is inherent to the feature and not dependent on a model).

#### 14. What is mutual information and KL divergence?

Mutual information is a symmetric measure, i.e. I(X;Y)=I(Y;X). However KL-Distance is an asymmetric measure: D(P||Q) not equal D(Q||P).

#### 15. Describe KL divergence intuitively.

Very often in Probability and Statistics we'll replace observed data or a complex distributions with a simpler, approximating distribution. KL Divergence helps us to measure just how much information we lose when we choose an approximation.

## Support Vector Machine

#### 1. How can the SVM optimization function be derived from the logistic regression optimization function?

SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.

#### 2. What is a large margin classifier?

The Maximal-Margin Classifier is a hypothetical classifier that best explains how SVM works in practice.

#### 3. Why SVM is an example of a large margin classifier? 

It maximizes the margin of the hyperplane. This is the best hyperplane because it reduces the generalization error the most. If we add new data, the Maximum Margin Classifier is the best hyperplane to correctly classify the new data.

#### 4. SVM being a large margin classifier, is it influenced by outliers?

SVM is not very robust to outliers. Presence of a few outliers can lead to very bad global misclassification. SVM is not very robust to outliers. 

#### 5. What is the role of C in SVM?

C is a regularization parameter that controls the trade off between the achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data. 

#### 6. In SVM, what is the angle between the decision boundary and theta?

The vector theta has to be perpendicular to decision boundary.. Because the goal here is to maximize the margin. ... To understand this think as if you want to calculate the distance of point from line1 which can be done by projecting the point on to the line perpendicular to line1.

#### 7. What is the mathematical intuition of a large margin classifier?

It maximizes the margin of the hyperplane. This is the best hyperplane because it reduces the generalization error the most. If we add new data, the Maximum Margin Classifier is the best hyperplane to correctly classify the new data.

#### 8. What is a kernel in SVM? Why do we use kernels in SVM?

SVM algorithms use a set of mathematical functions that are defined as the kernel. The function of kernel is to take data as input and transform it into the required form. These functions can be different types. For example linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.

#### 9. What is a similarity function in SVM? Why it is named so?

Given two vectors, the similarity is the length of the projection of one vector on another. Another interesting kernel examples is Gaussian kernel.
 
#### 10. How are the landmarks initially chosen in an SVM? How many and where?

#### 11. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?

The kernel trick avoids the explicit mapping that is needed to get linear learning algorithms to learn a nonlinear function or decision boundary. For all and in the input space , certain functions can be expressed as an inner product in another space . The function is often referred to as a kernel or a kernel function.

#### 12. What is the difference between logistic regression and SVM without a kernel?

SVM tries to find the widest possible separating margin, while Logistic Regression optimizes the log likelihood function, with probabilities modeled by the sigmoid function.

#### 13. How does the SVM parameter C affect the bias/variance trade off?

The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

#### 14. How does the SVM kernel parameter sigma² affect the bias/variance trade off?

#### 15. Can any similarity function be used for SVM?

The similarity is the length of the projection of one vector on another. Another interesting kernel examples is Gaussian kernel.

#### 16. Logistic regression vs. SVMs: When to use which one?

So, you can typically expect SVM to perform marginally better than logistic regression. SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.

## Bayesian Machine Learning

#### 1. What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?

The key difference between Bayesian and frequentist approaches lies in the definition of a probability, so if it is necessary to treat probabilties strictly as a long run frequency then frequentist approaches are reasonable, if it isn't then you should use a Bayesian approach.

#### 2. Compare and contrast maximum likelihood and maximum a posteriori estimation.

MLE gives you the value which maximises the Likelihood P(D|θ). And MAP gives you the value which maximises the posterior probability P(θ|D).

#### 3. How does Bayesian methods do automatic feature selection?

#### 4. What do you mean by Bayesian regularization?

Bayesian regularization is a mathematical process that converts a nonlinear regression into a "well-posed" statistical problem in the manner of a ridge regression.

#### 5. When will you use Bayesian methods instead of Frequentist methods?

1. Use of priors
2. Bayesian analysis tells us both how likely version A is to be the winner and by how much
3. Bayesian analysis can be more robust to outliers, by using more flexible distributions
4. Bayesian analysis does not require to pick a bunch of thresholds in advance to formulate a valid hypothesis

## Regularization

#### 1. What is L1 regularization?

The L1 regularization adds a penalty equal to the sum of the absolute value of the coefficients. The L1 regularization will shrink some parameters to zero. Hence some variables will not play any role in the model, L1 regression can be seen as a way to select features in a model.

#### 2. What is L2 regularization?

The L2 regularization adds a penalty equal to the sum of the squared value of the coefficients. The L2 regularization will force the parameters to be relatively small, the bigger the penalization, the smaller (and the more robust) the coefficients are.

#### 3. Compare L1 and L2 regularization.

The difference between the L1 and L2 is just that L2 is the sum of the square of the weights, while L1 is just the sum of the weights. L1 helps perform feature selection in sparse feature spaces.Feature selection is to know which features are helpful and which are redundant. L2 has one very important advantage to L1, and that is invariance to rotation and scale. This is especially important in geographical / physical application.

#### 4. Why does L1 regularization result in sparse models?

The reason for using L1 norm to find a sparse solution is due to its special shape. It has spikes that happen to be at sparse points. Using it to touch the solution surface will very likely to find a touch point on a spike tip and thus a sparse solution.

#### 5. What is dropout?

Dropout is a technique used to prevent a model from overfitting. Dropout works by randomly setting the outgoing edges of hidden units (neurons that make up hidden layers) to 0 at each update of the training phase.

#### 6. How will you implement dropout during forward and backward pass?

In forward propagation, inputs are set to zero with probability 𝑝, and otherwise scaled up by 11−𝑝.
In backward propagation, gradients for the same dropped units are zeroed out; other gradients are scaled up by the same 11−𝑝.
 
## Evaluation of Machine Learning systems

#### 1. What are accuracy, sensitivity, specificity, ROC?

Sensitivity, Specificity, and Accuracy, ROC are the terms which are most commonly associated with a Binary classification test and they statistically measure the performance of the test. 

#### 2. What are precision and recall?

Precision can be seen as a measure of exactness or quality, whereas recall is a measure of completeness or quantity. In simple terms, high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.

#### 3. Describe t-test in the context of Machine Learning.

A t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features.

## Clustering

#### 1. Describe the k-means algorithm.

K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided.

#### 2. What is distortion function? Is it convex or non-convex?

A non-convex optimization problem is any problem where the objective or any of the constraints are non-convex, as pictured below. Such a problem may have multiple feasible regions and multiple locally optimal points within each region.

#### 3. Tell me about the convergence of the distortion function.

#### 4. Topic: EM algorithm

The EM algorithm is a general procedure to estimate the parameters in a model with latent (unobserved) factors. 

#### 5. What is the Gaussian Mixture Model?

Gaussian mixture models are a probabilistic model for representing normally distributed subpopulations within an overall population. 

#### 6. Describe the EM algorithm intuitively.

It works by choosing random values for the missing data points, and using those guesses to estimate a second set of data. The new values are used to create a better guess for the first set, and the process continues until the algorithm converges on a fixed point.

#### 7. What are the two steps of the EM algorithm

E-step: perform probabilistic assignments of each data point to some class based on the current hypothesis h for the distributional class parameters;
M-step: update the hypothesis h for the distributional class parameters based on the new data assignments.

## Dimensionality Reduction

#### 1. Why do we need dimensionality reduction techniques?

It reduces the time and storage space required. Removal of multi-collinearity improves the interpretation of the parameters of the machine learning model. It becomes easier to visualize the data when reduced to very low dimensions such as 2D or 3D.

#### 2. What do we need PCA and what does it do?

Principal Component Analysis (PCA) is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.

#### 3. What is the difference between logistic regression and PCA?

PCA will NOT consider the response variable but only the variance of the independent variables.
Logistic Regression will consider how each independent variable impact on response variable.






## Basics of Probability and Information Theory

#### 1. Compare “Frequentist probability” vs. “Bayesian probability”?

Frequentists use probability only to model certain processes broadly described as "sampling." Bayesians use probability more widely to model both sampling and other kinds of uncertainty.

#### 2. What is a random variable?

A random variable is a variable whose value is unknown or a function that assigns values to each of an experiment's outcomes. 

#### 3. What is a probability distribution

A probability distribution is a table or an equation that links each outcome of a statistical experiment with its probability of occurrence.

#### 4. What is a probability mass function?

A probability mass function (pmf) is a function used to describe the probability associated with the discrete variable.

#### 5. What is a probability density function?

Probability density function (PDF) is a statistical expression that defines a probability distribution (the likelihood of an outcome) for a discrete random variable (e.g., a stock or ETF).

#### 6. What is a joint probability distribution?

A joint distribution is a probability distribution having two or more independent random variables. Furthermore, the strength of any relationship between the two variables can be measured.

#### 7. What are the conditions for a function to be a probability mass function?

All probabilities are positive: fx(x) ≥ 0.
Any event in the distribution (e.g. “scoring between 20 and 30”) has a probability of happening of between 0 and 1 (e.g. 0% and 100%).

#### 8. What are the conditions for a function to be a probability density function?

The absolute likelihood for a continuous random variable to take on any particular value is 0 (since there are an infinite set of possible values to begin with), the value of the PDF at two different samples can be used to infer, in any particular draw of the random variable, how much more likely it is that the random variable would equal one sample compared to the other sample.

#### 9. What is a marginal probability? Given the joint probability function, how will you calculate it?

Joint probability is the probability of two events occurring simultaneously. Marginal probability is the probability of an event irrespective of the outcome of another variable. 

#### 10. What is conditional probability? Given the joint probability function, how will you calculate it?

Here, P(A given B) is the probability of event A given that event B has occurred, called the conditional probability, described below. The joint probability is symmetrical, meaning that P(A and B) is the same as P(B and A). 

#### 11. State the Chain rule of conditional probabilities.

The chain rule permits the calculation of any member of the joint distribution of a set of random variables using only conditional probabilities.

#### 12. What are the conditions for independence and conditional independence of two random variables?

If two random variables, X and Y, are independent, they satisfy the following conditions. P(x|y) = P(x), for all values of X and Y. P(x ∩ y) = P(x) * P(y), for all values of X and Y.

#### 13. What are expectation, variance and covariance?

The expectation describes the average value and the variance describes the spread (amount of variability) around the expectation. Covariance refers to the measure of the directional relationship between two random variables.

#### 14. Compare covariance and independence.

Covariance can be positive, zero, or negative. ... If X and Y are independent variables, then their covariance is 0: Cov(X, Y ) = E(XY ) − µXµY = E(X)E(Y ) − µXµY = 0 The converse, however, is not always true.

#### 15. What is the covariance for a vector of random variables?

A covariance matrix is a square matrix giving the covariance between each pair of elements of a given random vector. 

#### 16. What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?

The Bernoulli distribution is a discrete distribution having two possible outcomes labelled by and in which ("success") occurs with probability and ("failure") occurs with probability.

#### 17. What is a multinoulli distribution?

It is a discrete probability distribution that describes the possible results of a random variable that can take on one of K possible categories, with the probability of each category separately specified. 

#### 18. What is a normal distribution?

The normal distribution is a probability function that describes how the values of a variable are distributed. It is a symmetric distribution where most of the observations cluster around the central peak and the probabilities for values further away from the mean taper off equally in both directions.

#### 19. Why is the normal distribution a default choice for a prior over a set of real numbers?

There is no reason to use the Normal distribution as a default prior for a real parameter.

#### 20. What is the central limit theorem?

The central limit theorem states that if you have a population with mean μ and standard deviation σ and take sufficiently large random samples from the population with replacement , then the distribution of the sample means will be approximately normally distributed.

#### 21. What are exponential and Laplace distribution?

It is the distribution of differences between two independent variates with identical exponential distributions.

#### 22. What are Dirac distribution and Empirical distribution?

It is used to model the density of an idealized point mass or point charge as a function equal to zero everywhere except for zero and whose integral over the entire real line is equal to one.

#### 23. What is mixture of distributions?

A mixture distribution is the probability distribution of a random variable that is derived from a collection of other random variables as follows: first, a random variable is selected by chance from the collection according to given probabilities of selection, and then the value of the selected random variable is realized. 

#### 24. Name two common examples of mixture of distributions?

Empirical and Gaussian Mixture

#### 25. Is Gaussian mixture model a universal approximator of densities?

#### 26. Write the formulae for logistic and softplus function.

#### 27. Write the formulae for Bayes rule.

#### 28. What do you mean by measure zero and almost everywhere?

A property holds almost everywhere if it holds for all elements in a set except a subset of measure zero, or equivalently, if the set of elements for which the property holds is conull.

#### 29. If two random variables are related in a deterministic way, how are the PDFs related?

#### 30. Define self-information. What are its units?

Self-information is defined as the amount of information that knowledge about (the outcome of) a certain event, adds to someone's overall knowledge. The amount of self-information is expressed in the unit of information: a bit.

#### 31. What are Shannon entropy and differential entropy?

Differential entropy (also referred to as continuous entropy) is a concept in information theory that began as an attempt by Shannon to extend the idea of (Shannon) entropy, a measure of average surprisal of a random variable, to continuous probability distributions.

#### 32. What is Kullback-Leibler (KL) divergence?

Kullback–Leibler divergence is a measure of how one probability distribution is different from a second, reference probability distribution.

#### 33. Can KL divergence be used as a distance measure?

Although the KL divergence measures the “distance” between two distri- butions, it is not a distance measure. This is because that the KL divergence is not a metric measure. It is not symmetric: the KL from p(x) to q(x) is generally not the same as the KL from q(x) to p(x).

#### 34. Define cross-entropy.

Cross-entropy is commonly used in machine learning as a loss function. Cross-entropy is a measure from the field of information theory, building upon entropy and generally calculating the difference between two probability distributions.

#### 35. What are structured probabilistic models or graphical models?

A structured probabilistic model is a probabilistic model for which a graph expresses the conditional dependence structure between random variables.

## Confidence interval

#### 1. What is population mean and sample mean?

Sample Mean is the mean of sample values collected. Population Mean is the mean of all the values in the population. If the sample is random and sample size is large then the sample mean would be a good estimate of the population mean.

#### 2. What is population standard deviation and sample standard deviation?

#### 3. Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? 

The formula we use for standard deviation depends on whether the data is being considered a population of its own, or the data is a sample representing a larger population.
If the data is being considered a population on its own, we divide by the number of data points, NNN.
If the data is a sample from a larger population, we divide by one fewer than the number of data points in the sample, n-1n−1n, minus, 1.

#### 4. What is the formula for calculating the s.d. of the sample mean?

Step 1: Calculate the mean of the data
Step 2: Subtract the mean from each data point. 
Step 3: Square each deviation to make it positive.
Step 4: Add the squared deviations together.
Step 5: Divide the sum by one less than the number of data points in the sample.

#### 5. What is confidence interval?

A Confidence interval is a range of values that likely would contain an unknown population parameter. Confidence level refers to the percentage of probability, or certainty, that the confidence interval would contain the true population parameter when you draw a random sample many times.

#### 6. What is standard error?

The standard error is a statistical term that measures the accuracy with which a sample distribution represents a population by using standard deviation.



## Curse of dimensionality

#### 1. The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions).

#### 2. What is local constancy or smoothness prior or regularization?

In order to generalize well, machine learning algorithms need to be guided by prior beliefs about what kind of function they should learn. This prior states that the function we learn should not change very much within a small region.





## Autoencoders

#### 1. What is an Autoencoder? What does it “auto-encode”?

Autoencoder is used for reducing the dimensions of the dataset while learning how to ignore noise. An Autoencoder is an unsupervised artificial neural network used for learning.

#### 2. What were Autoencoders traditionally used for? Why there has been a resurgence of Autoencoders for generative modeling?

An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. They are a self-supervised technique for representation learning, where our network learns about its input so that it may generate new data just as input.

#### 3. What is recirculation?

#### 4. What loss functions are used for Autoencoders?

Cross Entropy, KL divergence

#### 5. What is a linear autoencoder? Can it be optimal (lowest training reconstruction error)? If yes, under what conditions?

#### 6. What is the difference between Autoencoders and PCA?

PCA is essentially a linear transformation but Auto-encoders are capable of modelling complex non linear functions. PCA is faster and computationally cheaper than autoencoders.

#### 7. What is the impact of the size of the hidden layer in Autoencoders?

The goal of an autoencoder is to learn a good representation of the input which means that it should not be simply learning the identity function. One way to enforce this is by making the number of hidden units less than the number of features. There are also other constraints that work very well in practice, which allow you to have more hidden units than features potentially (Sparse Autoencoder, Denoising Autoencoder, Contractive Autoencoder, etc)

#### 8. What is an undercomplete Autoencoder? Why is it typically used for?




## Representation Learning

#### 1. What is representation learning? Why is it useful?

Representation learning is learning representations of input data typically by transforming it, that makes it easier to perform a task like classification or prediction.

#### 2. What is the relation between Representation Learning and Deep Learning?

In deep learning, the representations are formed by composition of multiple non-linear transformations of the input data with the goal of yielding abstract and useful representations for tasks like classification, prediction etc.

#### 3. What is one-shot and zero-shot learning (Google’s NMT)? Give examples.

In one shot learning, you get only 1 or a few training examples in some categories. In zero shot learning, you are not presented with every class label in training. So in some categories, you get 0 training examples.

#### 4. What trade offs does representation learning have to consider?

#### 5. What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?

Greedy Layer-Wise Unsupervised Pretraining. ... Greedy layer-wise pretraining is called so because it optimizes each layer at a time greedily. After unsupervised training, there is usually a fine-tune stage, when a joint supervised training algorithm is applied to all the layers.

#### 6. What were/are the purposes of the above technique? (deep learning problem and initialization)

Greedy layer-wise pretraining provides a way to develop deep multi-layered neural networks whilst only ever training shallow networks. Pretraining can be used to iteratively deepen a supervised model or an unsupervised model that can be repurposed as a supervised model.

#### 7. Why does unsupervised pretraining work?

Unsupervised pre-training initializes the model to a point in parameter space that somehow renders the optimization process more effective, in the sense of achieving a lower minimum of the empirical cost function.

#### 8. When does unsupervised training work? Under which circumstances?

In unsupervised learning, an AI system is presented with unlabeled, uncategorized data and the system's algorithms act on the data without prior training. The output is dependent upon the coded algorithms. Unsupervised learning algorithms can perform more complex processing tasks than supervised learning systems.

#### 9. Why might unsupervised pretraining act as a regularizer?

Unsupervised pre-training sets the parameter in a region from which better basins of attraction can be reached, in terms of generalization. Hence, although unsupervised pre-training is a regularizer, it can have a positive effect on the training objective when the number of training examples is large.

#### 10. What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?

You cannot get precise information regarding data sorting, and the output as data used in unsupervised learning is labeled and not known.
Less accuracy of the results is because the input data is not known and not labeled by people in advance.

#### 11. How do you control the regularizing effect of unsupervised pretraining?

Unsupervised pre-training tries to learn a representation g(X) from X. g(X) might result in a sparse representation of the data (e.g. PCA, non-linear dimensionality reduction etc). As a result, the prediction function f(g(X)) now is further restricted.

## Monte Carlo Methods

#### 1. What are deterministic algorithms?

A deterministic algorithm is an algorithm which, given a particular input, will always produce the same output, with the underlying machine always passing through the same sequence of states.

#### 2. What are Las vegas algorithms?

Las Vegas algorithm is a randomized algorithm that always gives correct results; that is, it always produces the correct result or it informs about the failure.

#### 3. What are deterministic approximate algorithms?

An approximate algorithm is a way of dealing with NP-completeness for optimization problem. This technique does not guarantee the best solution. The goal of an approximation algorithm is to come as close as possible to the optimum value in a reasonable amount of time which is at most polynomial time.

#### 4. What are Monte Carlo algorithms?

Monte Carlo algorithm is a randomized algorithm whose output may be incorrect with a certain (typically small) probability.



