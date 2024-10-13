from scipy.sparse import csr_matrix
import numpy as np

h=1/3

#Add on 3a:
#for room 1, 3 geom_room is n**2, for room 4 geom_room would be 2*(n/2)

n = 4 # CAUTION: when considering room 4, n needs to be an odd number!
h = 1/3
def sparse_matrix_smallroom(h, room_4=True):
    """
    creates a sparse matrix for room 1, 3, 4

    Paramaters:
    h: float
        mesh width
    room_4: boolean
        if True then it returns the matrix for room 4, if false it returns the matrix of room 1, 3
    
    Returns:
    out: scipy sparse csr_matrix
        Sparse matrix for the room 1, 3 or 4
    """
    # initiate lists for sparse matrix command
    row = []
    col = []
    value = []

    if room_4:
        n = int((1/h + 2)/2)
    else:
        n = int(1/h + 1)

    for k in range(n**2):
        i = k % n
        j = k//n # results in an integer

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

    # convert lists to arrays (csr_matrix command needs np.arrays as input)
    row = np.array(row)
    col = np.array(col)
    value = np.array(value)
    return csr_matrix((value, (row, col) )) * 1/(h**2)


def sparse_matrix_bigroom(h):
    """
    creates a sparse matrix for room 2

    Paramaters:
    h: float
        mesh width
    
    Returns:
    out: scipy sparse csr_matrix
        Sparse matrix for the room 2
    """
    # initiate lists for sparse matrix command
    row = []
    col = []
    value = []

    n = int(1/h + 1)

    for k in range(n*(2*n-1)):
        i = k % n
        j = k//n #results in an integer

        # lower neighbour
        if j > 0:
            if not ((j == 1) and (i == 0 or (i > 1 and i < n-1))) or (j == n and i == n-1):
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
            if not ((j == n-2 and i == 0) or (j == (2*n-3) and ((i > 1 and i < n-2) or (i == n-1)))):             
                row.append(k)
                col.append(k + n)
                value.append(1)

    # convert lists to arrays (csr_matrix command needs np.arrays as input)
    row = np.array(row)
    col = np.array(col)
    value = np.array(value)
    return csr_matrix((value, (row, col) )) * 1/(h**2)