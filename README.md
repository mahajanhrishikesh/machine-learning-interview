# Machine Learning Interview Questions and Answers

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

5. If inverse of a matrix exists, how to calculate it

Step 1: calculating the Matrix of Minors,
Step 2: then turn that into the Matrix of Cofactors,
Step 3: then the Adjugate, and
Step 4: multiply that by 1/Determinant.

6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?

The determinant is a scalar value that can be computed from the elements of a square matrix and encodes certain properties of the linear transformation described by the matrix. 

Pick any row or column in the matrix.
Multiply every element in that row or column by its cofactor and add. The result is the determinant.

If A is an n × n matrix, then the sum of the n eigenvalues of A is the trace of A and the product of the n eigenvalues is the determinant of A.
