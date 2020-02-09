# Machine Learning Interview

## Machine learning interview questions and answers 

This list was compiled while preparing for AI Residency programs in Google, Facebook, Microsoft, Uber etc.

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
