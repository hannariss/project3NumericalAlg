from sparse_matrix import sparse_matrix_smallroom, sparse_matrix_bigroom
import DNmethod as dn

# Initialize variables

h = 1/20 # Size of steps (delta x)

A1 = sparse_matrix_smallroom(h, room_4=False) # Coefficient matrix A for rooms 1 and 3
A2 = sparse_matrix_bigroom(h) # Coefficient matrix A for room 2

# Create flat object
apartment = dn.DN_Method(h, t_n=15, t_h=40)

# # Dirichlet Neumann Iteration40
apartment.dn_iteration(A1, A2)

## Project 3a:

# A3 = sparse_matrix_smallroom(4, h, 16) Coefficient matrix A for room 4

# # Create flat object
# apartment = dn.DN_Method4Rooms(h)

# # Dirichlet Neumann Iteration
# apartment.dn_iteration(A1, A2, A3)