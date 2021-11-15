# mlib

mlib (mathlib) is a collection of math based python libraries meant for educational purposes. These are not production libraries.

# matrix.py

## What is it

An implementation of matrix, vectors and common in python. 

## API

### Vectors

```python

import matrix

column_vector = [
          [1]
        , [2]
        , [3]
]

column_vector_2 = [
      [1]
    , [1]
    , [1]
]

# Create
u = matrix.Matrix(column_vector)
v = matrix.Matrix(column_vector_2)

# Dimension
u.dimension

# Transpose to row vector
u.transpose()

# Iterate
for component in u:
    print(component)

# Sum 
u + v 

# Subtract
u + (-1 * v)
u - v 

# Scale
3 * u 

# Linear combinations
3 * u + v 

# TODO: implement this 
u.linear_combination


# Length
u.length

# Dot product
matrix.dot(u, v)

# Note: equivalent operations available on row vectors

```

### Matrices

```python

import matrix

column_vector = [
     [2]
   , [4]
]
reflection_matrix = [
    [0, 1],
    [1, 0],
]

# Create
A = matrix.Matrix(reflection_matrix)

# Dimension
A.dimension

# Multiply
u = matrix.Matrix(column_vector)
u = A * u

# Create matrix from vector func
def vector_func(x_0, x_1, x_2) -> [[]]:
    return [
          [x_0 + x_1 + x_2]
        , [x_0 + 2 * x_2]
        , [x_1 + 2 * x_2]
    ]

A = matrix.matrix_from_vector_func(vector_func)

```

