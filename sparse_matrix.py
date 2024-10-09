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
k = np.arange(0, n**2, 1)

row = []
col = []
value = []

for k in range(len(k)):
    i = k % n
    j = k//n #results in an integer
    
    # Linker Nachbar
    if i > 0:
        row.append(k)
        col.append(k-1)
        value.append(1)

    #Diagonale
    row.append(k)
    col.append(k)
    #je nachdem welches k wir betrachten muss eine -3 oder -4 eingetragen werden
    if i == (n - 1):
        value.append(-3)
    else:
        value.append(-4)

    # Rechter Nachbar
    if i < n - 1:
        row.append(k)
        col.append(k + 1)
        value.append(1)

    # Oberer Nachbar
    if j < n-1:
        row.append(k)
        col.append(k + n)
        value.append(1)

    # Unterer Nachbar
    if j > 0:
        row.append(k)
        col.append(k-n)
        value.append(1)

row = np.array(row)
col = np.array(col)
value = np.array(value)

A = csr_matrix((value, (row, col) )) * 1/(h**2)
