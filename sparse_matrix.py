from scipy.sparse import csr_matrix
import numpy as np

h=1/3

A1 = 1/(h**2) * np.array([[-4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                          [1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, -3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 1, -3, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -3, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -3]])

A = csr_matrix(A1)

# Sparse matrix: row_ind, col_ind, value

row_ind = np.array([0, 0, 0])
col_ind = np.array([0, 1, 4]) 
value = np.array([-4, 1, 1])

n = 4

def sparse_matrix_smallroom(n):
    k = np.arange(0, n**2, 1)

    row = []
    col = []
    value = []

    for k in range(len(k)):
        i = k % n
        j = k//n #results in an integer

        # lower neighbour
        if j > 0:
            if not ((i > 1 and i < n-1) and j == 1):
                row.append(k)
                col.append(k-n)
                value.append(1)
        
        # left neighbour
        if i > 0:
            if not (i == 1 and (j > 0 and j < n-1)): # using dirichlet for all walls makes us loose the left neighbour in some cases
                row.append(k)
                col.append(k-1)
                value.append(1)

        # diagonal
        row.append(k)
        col.append(k)
        # according to which k we are looking at it must be stored a -3 or -4
        if i == (n - 1):
            value.append(-3)
        else:
            value.append(-4)

        # right neighbour
        if i < n - 1:
            row.append(k)
            col.append(k + 1)
            value.append(1)

        # upper neighbour
        if j < n-1:
            if not ((i > 1 and i < n-1) and j == n-2):
                row.append(k)
                col.append(k + n)
                value.append(1)

    row = np.array(row)
    col = np.array(col)
    value = np.array(value)
    return csr_matrix((value, (row, col) )) * 1/(h**2)


def sparse_matrix_bigroom(n):
    row = []
    col = []
    value = []

    for k in range(n*(2*n-1)):
        i = k % n
        j = k//n #results in an integer

        # lower neighbour
        if j > 0:
            if not ((j == 1) and (i == 0 or i == n-2)) or (j == n and i == n-1):
                row.append(k)
                col.append(k-n)
                value.append(1)
        
        # left neighbour
        if i > 0:
            if not (i == 1 and j < (2*n-3)):
                row.append(k)
                col.append(k-1)
                value.append(1)

        # diagonal
        row.append(k)
        col.append(k)
        value.append(-4)

        # right neighbour
        if i < n - 1:
            if not (i == n-2 and j > 1 ):
                row.append(k)
                col.append(k + 1)
                value.append(1)

        # upper neighbour
        if j < (2*n-2):
            if not (j == n-2 and i == 0) or (j == (2*n-2) and (i == 1 or i == n-1)):
                row.append(k)
                col.append(k + n)
                value.append(1)

    row = np.array(row)
    col = np.array(col)
    value = np.array(value)
    return csr_matrix((value, (row, col) )) * 1/(h**2)
