# Linear Algebra for AI/ML

*Linear algebra is one of the most important parts of mathematics in machine learning. It helps us represent data, perform transformations, and understand how models work internally. This file covers the basic ideas of scalars, vectors, matrices, operations on them, and why they matter in AI/ML.*

- In simple words, linear algebra is the mathematics of **numbers arranged in lists and tables**.

- In machine learning, data is usually stored in the form of **vectors** and **matrices**.

- If calculus helps us understand change, then linear algebra helps us understand **structure and representation**.

- Almost every ML algorithm uses linear algebra in some way:
  - datasets are matrices
  - features are vectors
  - model weights are vectors
  - transformations are matrix operations

- So if we understand linear algebra clearly, it becomes much easier to understand machine learning.

## Scalar, Vector, and Matrix

- A **scalar** is just a single number.

Examples:
- 5
- -2
- 10.7

- A **vector** is a list of numbers arranged in order.

Example:

$$
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

- A vector can represent many things in ML, such as:
  - one data point
  - one group of features
  - one set of model weights

- A **matrix** is a rectangular table of numbers.

Example:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

- A matrix can represent:
  - a dataset
  - image pixels
  - a transformation
  - relationships between variables

- So:
  - scalar = one number
  - vector = one-dimensional collection of numbers
  - matrix = two-dimensional collection of numbers

## Shape and Dimension

- The **shape** tells us how many rows and columns an object has.

Examples:
- A vector with 3 values has shape `(3,)`
- A matrix with 2 rows and 3 columns has shape `(2, 3)`

- In ML, checking shape is very important because many operations only work when dimensions match correctly.

- For example:
  - two vectors can be added only if they have the same size
  - matrix multiplication works only when inner dimensions match

## Vector Addition

If two vectors have the same size, we add them element by element.

Example:

a = [1, 2, 3]  
b = [4, 5, 6]  

Result:

[1+4, 2+5, 3+6] = [5, 7, 9]

- This is useful when combining information or updating parameters.

## Scalar Multiplication

A vector can be multiplied by a single number (scalar).

Example:

2 × [1, 2, 3] = [2, 4, 6]

This simply multiplies each element of the vector.

- This changes the size of the vector but not its basic direction if the scalar is positive.

- In ML, scalar multiplication appears when applying learning rates and scaling values.

## Dot Product

- The **dot product** is one of the most important operations in linear algebra.

For two vectors:

$$
a = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix}, \quad
b = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
$$

their dot product is:

$$
a \cdot b = a_1b_1 + a_2b_2 + a_3b_3
$$

Example:

$$
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
\cdot
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
= 1\cdot4 + 2\cdot5 + 3\cdot6 = 32
$$

- The dot product measures how much two vectors point in a similar direction.

- In ML, dot product is everywhere:
  - linear regression
  - logistic regression
  - neural networks
  - similarity calculations

## Matrix Addition

Two matrices can be added if they have the same shape.

Example:

Matrix A:
[1  2]  
[3  4]

Matrix B:
[5  6]  
[7  8]

Result:

[1+5   2+6]  
[3+7   4+8]

=  

[6   8]  
[10  12]

- This is just element-by-element addition.

## Matrix Multiplication

Matrix multiplication is not element-wise.  
We multiply rows of the first matrix with columns of the second.

Example:

Matrix A:
[1  2]  
[3  4]

Matrix B:
[5  6]  
[7  8]

Result:

First row:
(1×5 + 2×7) = 19  
(1×6 + 2×8) = 22  

Second row:
(3×5 + 4×7) = 43  
(3×6 + 4×8) = 50  

Final matrix:

[19  22]  
[43  50]
- In machine learning, matrix multiplication is extremely important because models use it to transform input data into predictions.

## Transpose of a Matrix

- The **transpose** of a matrix means converting rows into columns and columns into rows.

If

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

then

$$
A^T =
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
$$

- Transpose is used in many formulas in ML, especially in optimization and matrix operations.

## Identity Matrix

- The **identity matrix** is a special square matrix with:
  - 1 on the main diagonal
  - 0 everywhere else

Example:

$$
I =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

- It behaves like the number 1 in matrix multiplication.

$$
AI = A
$$

- Identity matrices are used in many mathematical operations, including inverses and regularization.

## Inverse of a Matrix

- The inverse of a matrix is similar to reciprocal in normal arithmetic.

If $A^{-1}$ is the inverse of $A$, then:

$$
AA^{-1} = I
$$

- Not every matrix has an inverse.
- Only square matrices with non-zero determinant can have an inverse.

- Inverse is useful in solving systems of equations and appears in some ML formulas.

## Determinant

- The determinant is a number associated with a square matrix.

For a 2x2 matrix:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

the determinant is:

$$
\det(A) = ad - bc
$$

- If determinant is 0, the matrix is singular, which means it cannot be inverted.

- Determinant helps us understand:
  - invertibility
  - scaling effect
  - dependency between rows/columns

## Linear Algebra in Machine Learning

- In machine learning:
  - one data sample is usually a vector
  - a full dataset is usually a matrix
  - weights are vectors
  - predictions are often made using dot products and matrix multiplication

- Example:
  - if a student has features like age, study hours, and attendance, those features can be written as a vector
  - if we collect this data for many students, the full data becomes a matrix

- So linear algebra is not just theory. It is the actual language in which machine learning data is written.

## Key Takeaways

- Scalar = one number
- Vector = ordered list of numbers
- Matrix = rectangular table of numbers
- Shapes must match for valid operations
- Dot product is a core operation in ML
- Matrix multiplication is used in almost every model
- Transpose, determinant, identity, and inverse are foundational ideas

---

## Coding Tasks (use Colab or notebook)

```markdown
### 1. Create scalar, vector, and matrix

```python
import numpy as np

scalar = 5
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:\n", matrix)


```markdown
### 2. Perform vector addition and scalar multiplication

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector a:", a)
print("Vector b:", b)
print("Addition:", a + b)
print("Scalar multiplication (2 * a):", 2 * a)


```markdown
### 3. Find the dot product

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Dot product:", np.dot(a, b))


```markdown
### 4. Perform matrix addition and multiplication

```python
import numpy as np

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

print("Matrix m1:\n", m1)
print("Matrix m2:\n", m2)
print("Addition:\n", m1 + m2)
print("Multiplication:\n", np.dot(m1, m2))


```markdown
### 5. Transpose, determinant, and inverse

```python
import numpy as np

m = np.array([[1, 2], [3, 4]])

print("Original matrix:\n", m)
print("Transpose:\n", m.T)
print("Determinant:", np.linalg.det(m))
print("Inverse:\n", np.linalg.inv(m))


```markdown
### 6. Check shape of vector and matrix

```python
import numpy as np

vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4], [5, 6]])

print("Vector shape:", vector.shape)
print("Matrix shape:", matrix.shape)
