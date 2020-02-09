# Machine Learning Interview Questions and Answers

This list was compiled while preparing for AI Residency programs in Google, Facebook, Microsoft, Uber etc.

## Linear Algebra

1. What is broadcasting in connection to Linear Algebra

The term broadcasting describes how numpy treats arrays with 
different shapes during arithmetic operations. Subject to certain 
constraints, the smaller array is “broadcast” across the larger 
array so that they have compatible shapes. Broadcasting provides a 
means of vectorizing array operations so that looping occurs in C
instead of Python. It does this without making needless copies of 
data and usually leads to efficient algorithm implementations.

2. What are scalars, vectors, matrices, and tensors

Scalar: A single number.
Vector : A list of values.(rank 1 tensor)
Matrix: A two dimensional list of values.(rank 2 tensor)
Tensor: A multi dimensional matrix with rank n.

3. What is Hadamard product of two matrices

Hadamard product is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands where each element i, j is the product of elements i, j of the original two matrices.

4. What is an inverse matrix

The inverse of a matrix A is a matrix that, when multiplied by A results in the identity. Invertible matrices have connections back to systems of equations and to other concepts like linear independence or dependence.

5. If inverse of a matrix exists, how to calculate it

Step 1: calculating the Matrix of Minors,
Step 2: then turn that into the Matrix of Cofactors,
Step 3: then the Adjugate, and
Step 4: multiply that by 1/Determinant.

6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?

The determinant is a scalar value that can be computed from the elements of a square matrix and encodes certain properties of the linear transformation described by the matrix. Pick any row or column in the matrix. Multiply every element in that row or column by its cofactor and add. The result is the determinant. If A is an n × n matrix, then the sum of the n eigenvalues of A is the trace of A and the product of the n eigenvalues is the determinant of A.

7. Discuss span and linear dependence

If we use a linearly dependent set to construct a span, then we can always create the same infinite set with a starting set that is one vector smaller in size. However, this will not be possible if we build a span from a linearly independent set. So in a certain sense, using a linearly independent set to formulate a span is the best possible way — there are not any extra vectors being used to build up all the necessary linear combinations.

8. What is Ax = b? When does Ax =b has a unique solution

If and only if b is a linear combination of the columns of A

9. In Ax = b, what happens when A is fat or tall

The homogeneous system Ax = 0 has infinitely many solutions. The statement would be true for fat and singular
square matrices, but fails for a nonsingular square matrix.

10. When does inverse of A exist

Inverse of a matrix exists when the matrix is invertible. Now for a matrix to be invertible , you need to have the condition that the determinant of the matrix must not be zero. That is det(A) ≠ 0 where A is your matrix of interest.

11. What is a norm? What is L1, L2 and L infinity norm

L1 Norm is the sum of the magnitudes of the vectors in a space. It is the most natural way of measure distance between vectors, that is the sum of absolute difference of the components of the vectors. L2 norm gives the shortest distance to go from one point to another. L infinity norm gives the largest magnitude among each element of a vector.

12. What are the conditions a norm has to satisfy

If norm of x is greater than 0 then x is not equal to 0 (Zero Vector) and if norm is equal to 0 then x is a zero vector. 

13. Why is squared of L2 norm preferred in ML than just L2 norm

On raising power to cube (L3), quad (L4) or higher, the function becomes sensitive to the influence of outliers and thus introduces unwanted bias into distance calculation. Using squared L2 norm, the function becomes easily calculable.

14. When L1 norm is preferred over L2 norm

L1 is used for feature selection, dealing with sparsity and has less computational cost.

15. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?

Yes, although it is actually not a norm. It corresponds to the total number of nonzero elements in a vector.

16. What is Frobenius norm?

The Frobenius norm is matrix norm of a matrix defined as the square root of the sum of the absolute squares of its elements.

17. What is a diagonal matrix?

A matrix having non-zero elements only in the diagonal running from the upper left to the lower right.

18. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?

19. At what conditions does the inverse of a diagonal matrix exist?

The inverse of a diagonal matrix is obtained by replacing each element in the diagonal with its reciprocal, as illustrated below for matrix C. where I is the identity matrix. 

20. What is a symmetrix matrix?

A symmetric matrix is a square matrix that is equal to its transpose. 

21. What is a unit vector?

A vector which has a magnitude of one.

22. When are two vectors x and y orthogonal?

Two vectors are orthogonal if and only if their dot product is zero, i.e. they make an angle of 90° (π/2 radians), or one of the vectors is zero.

23. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?

Given several non-zero vectors, if they are orthogonal (one to another), then they are linearly independent. So their number cannot exceed 𝑛 if they are in ℝ𝑛 .

24. When are two vectors x and y orthonormal

To summarize, for a set of vectors to be orthogonal : They should be mutually perpendicular to each other

25. What is an orthogonal matrix? Why is computationally preferred?

An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (i.e., orthonormal vectors), i.e. where. is the identity matrix. Because it is invertible.

26. What is eigendecomposition, eigenvectors and eigenvalues?

Eigendecomposition of a matrix is a type of decomposition that involves decomposing a square matrix into a set of eigenvectors and eigenvalues. One of the most widely used kinds of matrix decomposition is called eigendecomposition, in which we decompose a matrix into a set of eigenvectors and eigenvalues. An eigenvector or characteristic vector of a linear transformation is a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it. The corresponding eigenvalue is the factor by which the eigenvector is scaled.

27. How to find eigen values of a matrix?





## Representation Learning

1. What is representation learning? Why is it useful?

Representation learning is learning representations of input data typically by transforming it, that makes it easier to perform a task like classification or prediction.

2. What is the relation between Representation Learning and Deep Learning?

In deep learning, the representations are formed by composition of multiple non-linear transformations of the input data with the goal of yielding abstract and useful representations for tasks like classification, prediction etc.

3. What is one-shot and zero-shot learning (Google’s NMT)? Give examples.

In one shot learning, you get only 1 or a few training examples in some categories. In zero shot learning, you are not presented with every class label in training. So in some categories, you get 0 training examples.

4. What trade offs does representation learning have to consider?

5. What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?

Greedy Layer-Wise Unsupervised Pretraining. ... Greedy layer-wise pretraining is called so because it optimizes each layer at a time greedily. After unsupervised training, there is usually a fine-tune stage, when a joint supervised training algorithm is applied to all the layers.

6. 





## Monte Carlo Methods

1. What are deterministic algorithms?

A deterministic algorithm is an algorithm which, given a particular input, will always produce the same output, with the underlying machine always passing through the same sequence of states.

2. What are Las vegas algorithms?

Las Vegas algorithm is a randomized algorithm that always gives correct results; that is, it always produces the correct result or it informs about the failure.

3. What are deterministic approximate algorithms?

An approximate algorithm is a way of dealing with NP-completeness for optimization problem. This technique does not guarantee the best solution. The goal of an approximation algorithm is to come as close as possible to the optimum value in a reasonable amount of time which is at most polynomial time.

4. What are Monte Carlo algorithms?

Monte Carlo algorithm is a randomized algorithm whose output may be incorrect with a certain (typically small) probability.



