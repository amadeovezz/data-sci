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
v = matrix.Matrix(column_vector)
u = matrix.Matrix(column_vector_2)

# Dimension
v.dimension

# Transpose to row vector
v.transpose()

# Iterate
for component in v:
    print(component)

# Sum 
v + u 

# Subtract
v + (-1 * u)
v - u 

# Scale
3 * v 

# Linear combinations
3 * v + u 

# TODO: implement this 
v.linear_combination


# Length
v.length

# Dot product
matrix.dot(v, u)

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
v = matrix.Matrix(column_vector)
u = A * v

# Create matrix from vector func
def vector_func(x_0, x_1, x_2) -> [[]]:
    return [
          [x_0 + x_1 + x_2]
        , [x_0 + 2 * x_2]
        , [x_1 + 2 * x_2]
    ]

A = matrix.matrix_from_vector_func(vector_func)

```

