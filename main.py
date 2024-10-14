from sparse_matrix import sparse_matrix_smallroom, sparse_matrix_bigroom
import DNmethod as dn

### Task 1:

# Initialize variables
h = 1/3 # Size of steps (delta x)
A1 = sparse_matrix_smallroom(h, room_4=False) # Coefficient matrix A for rooms 1 and 3
A2 = sparse_matrix_bigroom(h) # Coefficient matrix A for room 2

# Create flat object
apartment = dn.DN_Method(h)

# Dirichlet Neumann Iteration
apartment.dn_iteration(A1, A2)

### Task 2:
# Except in the lower parts of room 2 (close to the window), the temperature in all rooms is about 20 degrees.
# Assuming that 20 degrees is a reasonable temperature for winter, the heating in the flat is adequate.

### Task 3:

h = 1/20 
A1 = sparse_matrix_smallroom(h, room_4=False) # Coefficient matrix A for rooms 1 and 3
A2 = sparse_matrix_bigroom(h) # Coefficient matrix A for room 2

apartment = dn.DN_Method(h, t_n=15, t_h=40)
apartment.dn_iteration(A1, A2)


### Project 3a:

h = 1/100
A1 = sparse_matrix_smallroom(h, room_4=False) # Coefficient matrix A for rooms 1 and 3
A2 = sparse_matrix_bigroom(h) # Coefficient matrix A for room 2
A3 = sparse_matrix_smallroom(h) # Coefficient matrix A for room 4

apartment2 = dn.DN_Method4Rooms(h)
apartment2.dn_iteration(A1, A2, A3)