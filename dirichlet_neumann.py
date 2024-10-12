from mpi4py import MPI
import numpy as np
from scipy.linalg import solve

from sparse_matrix import sparse_matrix_smallroom, sparse_matrix_bigroom

#----------------------------
#    Initialize Variables
#----------------------------

# Initialize MPI communication
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
rank = comm.Get_rank()

# Size of steps (delta x) and relaxation parameter
h = 1/3
omega = 0.8

# Set wall temperatures 
t_n = 15 # normal walls
t_h = 40 # walls with heater 
t_w = 5 # wallls with window

# Initialize temperature vectors
def init_u(h, t_n=15, t_h=40, t_w=5, t_r=20):
    n = (1 / h) + 1
    u1_init = np.zeros(int(n ** 2))
    u2_init = np.zeros(int(n * ((2 * n) - 1)))
    u3_init = np.zeros(int(n ** 2))

    # Fill u1_init
    for ix in range(len(u1_init)):
        i = ix % n
        j = ix // n
        if (j == 0 or j == n - 1) and i != 0:
            u1_init[ix] = t_n
        elif i == 0:
            u1_init[ix] = t_h
        else:
            u1_init[ix] = t_r
    
    # Fill u2_init
    for ix in range(len(u2_init)):
        i = ix % n
        j = ix // n
        if j == 0 and i != 0:
            u2_init[ix] = t_w
        elif i == n - 1 and j != 0 and j < n:
            u2_init[ix] = t_n
        elif i == 0 and j != 2 * n - 2 and j >= n:
            u2_init[ix] = t_n
        elif j == 2 * n - 2 and i != n - 1:
            u2_init[ix] = t_h
        elif (j == 0 and i == 0) or (j == 2 * n - 2 and i == n - 1):
            u2_init[ix] = t_n
        else:
            u2_init[ix] = t_r
    
    # Fill u3_init
    for ix in range(len(u3_init)):
        i = ix % n
        j = ix // n
        if (j == 0 or j == n - 1) and i != 0:
            u3_init[ix] = t_n
        elif i == 0:
            u3_init[ix] = t_h
        else:
            u3_init[ix] = t_r

    return u1_init, u2_init, u3_init

u1_init, u2_init, u3_init = init_u(h)


# Coefficient matrix A for rooms 1 and 3
# A1 = 1/(h**2) * np.array([[-4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 1, -3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 1, 0, 0, 1, -3, 0, 0, 0, 1, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -4, 1, 0, 0, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -3, 0, 0, 0, 1],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -3]])

A1 = sparse_matrix_smallroom(n=4)

# Coefficient matrix A for room 2
# A2 = 1/(h**2) * np.array([[-4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4]])

A2 = sparse_matrix_bigroom(4)

#----------------------------
#         Methods
#----------------------------

def gamma(vec1, vec2=None): # Note: Specific for h = 1/3
    '''
    Derives boundary values (Gamma) from temperature vectors of the rooms.
    If two temperature vectors are provided, it extracts boundary values for small rooms (room 1 and room 3).
    If only one temperature vector is provided, it extracts boundary values for the large room (room 2).
    
    Parameters:
    vec1 (ndarray): Temperature vector of room 1 or 2.
    vec2 (ndarray, optional): Temperature vector of room 3.
    
    Returns:
    Gamma1 (ndarray): Boundary temperatures from one side of the room.
    Gamma2 (ndarray): Boundary temperatures from the opposite side of the room.
    '''
    if vec2 is not None: # Two temperature vectors have been provided
        Gamma1 = np.array([vec1[3], vec1[7], vec1[11], vec1[15]])
        Gamma2 = np.array([vec2[15], vec2[11], vec2[7], vec2[3]])
    else: # For room 2
        Gamma1 = np.array([vec1[0], vec1[4], vec1[8], vec1[12]]) 
        Gamma2 = np.array([vec1[15], vec1[19], vec1[23], vec1[27]])
    return Gamma1, Gamma2

# Step 1: Solve problem for room 2
def solve_omega2(Gamma1, Gamma2, A): # Note: Specific for h = 1/3
    '''
    Solves for the temperature distribution in room 2 given the boundary values (Gamma1 and Gamma2).

    Parameters:
    Gamma1 (ndarray): Boundary values from room 1.
    Gamma2 (ndarray): Boundary values from room 3.
    A (ndarray): Coefficient matrix for room 2.

    Returns:
    u2_new (ndarray): New temperature distribution in room 2.
    '''
    u_DC = - 1/(h**2) * np.array([0, Gamma1[0], 0, 0, 15, Gamma1[1], 5, 0, 15, Gamma1[2], 15, 0, 0, Gamma1[3], Gamma2[0], 0, 0, 15, Gamma2[1], 15, 0, 40, Gamma2[2], 15, 0, 0, Gamma2[3], 0])
    u2_new = solve(A, u_DC)
    return u2_new

# Step 2: Solve problem for room 1 and 3 (single functions for each room)
def solve_omega1(Gamma1, A): # Note: Specific for h = 1/3
    '''
    Solves for the temperature distribution in room 1 given the boundary values from room 2.

    Parameters:
    Gamma1 (ndarray): Boundary values from room 2.
    A (ndarray): Coefficient matrix for room 1.

    Returns:
    u1_new (ndarray): New temperature distribution in room 1.
    '''
    u_NC = - 1/h * np.array([0, 0, 0, Gamma1[0], 0, 40/h, 15/h, Gamma1[1], 0, 40/h, 15/h, Gamma1[2], 0, 0, 0, Gamma1[3]])
    u1_new = solve(A, u_NC)
    return u1_new

def solve_omega3(Gamma2, A): # Note: Specific for h = 1/3
    '''
    Solves for the temperature distribution in room 3 given the boundary values from room 2.

    Parameters:
    Gamma2 (ndarray): Boundary values from room 2.
    A (ndarray): Coefficient matrix for room 3.

    Returns:
    u3_new (ndarray): New temperature distribution in room 3.
    '''
    u_NC = - 1/h * np.array([0, 0, 0, Gamma2[3], 0, 40/h, 15/h, Gamma2[2], 0, 40/h, 15/h, Gamma2[1], 0, 0, 0, Gamma2[0]])
    u3_new = solve(A, u_NC)
    return u3_new

# Step 3: Relaxation
def relax(u, u_new):
    '''
    Applies relaxation method to smooth the temperature values by combining the old and new solutions.

    Parameters:
    u (ndarray): Current temperature vector.
    u_new (ndarray): New calculated temperature vector.

    Returns:
    ndarray: Relaxed temperature vector.
    '''
    u_new = omega * u_new + (1 - omega) * u
    return u_new

def reset_walls_13(u1, u3): # Note: Specific for h = 1/3
    '''
    Resets the wall temperatures to predefined values after solving for the temperature distribution in rooms 1 and 3.

    Parameters:
    u1 (ndarray): Temperature vector for room 1.
    u3 (ndarray): Temperature vector for room 3.

    Returns:
    tuple: Updated temperature vectors for room 1 and room 3 with boundary (wall) values reset.
    '''
    u1_walls = np.array([40, 15, 15, 15, 40, u1[5], u1[6], u1[7], 40, u1[9], u1[10], u1[11], 40, 15, 15, 15])
    u3_walls = np.array([40, 15, 15, 15, 40, u3[5], u3[6], u3[7], 40, u3[9], u3[10], u3[11], 40, 15, 15, 15])
    return u1_walls, u3_walls

def reset_walls_2(u2):  # Note: Specific for h = 1/3
    '''
    Resets the wall temperatures to predefined values after solving for the temperature distribution in room 2.

    Parameters:
    u2 (ndarray): Temperature vector for room 2.

    Returns:
    ndarray: Updated temperature vector for room 2 with boundary (wall) values reset.
    '''
    u2_walls = np.array([15, 5, 5, 5, u2[4], u2[5], u2[6], 15, u2[8], u2[9], u2[10], 15, 15, u2[13], u2[14], 15, 15, u2[17], u2[18], u2[19], 15, u2[21], u2[22], u2[23], 40, 40, 40, 15])
    return u2_walls

#----------------------------
#        Iteration
#----------------------------

u1 = u1_init
u2 = u2_init
u3 = u3_init

### Iteration process with MPI

# Print initial conditions in process 2
if rank == 2:
    print(f'\n Initial Conditions:')
    print(f'\n Temperature in Omega 1: \n {u1.reshape(4,4)} \n\n Temperature in Omega 2: \n {u2.reshape(7,4)} \n\n Temperature in Omega 3: \n {u3.reshape(4,4)}')

num_iterations = 10

for iteration in range(num_iterations):
    
    # Process 0 computes the temperature distribution for room 2 (Omega 2) in each iteration
    if rank == 0:
        # Receive Gamma1 and Gamma2 from process 1
        Gamma1 = np.empty(4)
        Gamma2 = np.empty(4)
        comm.Recv(Gamma1, source=1, tag=iteration)
        comm.Recv(Gamma2, source=1, tag=(iteration+10))

        # Compute u2_new
        u2_new = solve_omega2(Gamma1, Gamma2, A2)
        u2_new = reset_walls_2(u2_new)

        # Relaxation
        u2_new = relax(u2, u2_new)

        # Send Gamma1 and Gamma2 to process 1
        Gamma1, Gamma2 = gamma(u2_new)
        comm.Send(Gamma1, dest=1, tag=iteration)
        comm.Send(Gamma2, dest=1, tag=(iteration+10))

        # Send u2_new to process two
        comm.Send(u2_new, dest=2, tag=iteration)

        # Update u2
        u2 = u2_new  

    # Process 1 computes the temperature distributions for room 1 (Omega 1) and room 3 (Omega 3) in each iteration
    if rank == 1:
        # Send Gamma1 and Gamma2 to process 0
        Gamma1, Gamma2 = gamma(u1, u3)
        comm.Send(Gamma1, dest=0, tag=iteration)
        comm.Send(Gamma2, dest=0, tag=(iteration+10))

        # Receive Gamma1 and Gamma2 from process 0
        Gamma1 = np.empty(4)
        Gamma2 = np.empty(4)
        comm.Recv(Gamma1, source=0, tag=iteration)
        comm.Recv(Gamma2, source=0, tag=(iteration+10))

        # Compute u1_new and u3_new
        u1_new = solve_omega1(Gamma1, A1)
        u3_new = solve_omega3(Gamma2, A1)
        u1_new, u3_new = reset_walls_13(u1_new, u3_new)

        # Relaxation
        u1_new = relax(u1, u1_new)
        u3_new = relax(u3, u3_new)

        # Send u1_new and u3_new to process two
        comm.Send(u1_new, dest=2, tag=iteration)
        comm.Send(u3_new, dest=2, tag=(iteration+10))

        # Update u1 and u3
        u1 = u1_new
        u3 = u3_new

    # Process 2 prints the temperature distributions for each room in each iteration in form of matrices
    if rank == 2:
        # Receive new u1, u2 and u3 vectors from processes 0 and 1
        u1 = np.empty(16)
        u2 = np.empty(28)
        u3 = np.empty(16)     
        comm.Recv(u2, source=0, tag=iteration)
        comm.Recv(u1, source=1, tag=iteration)
        comm.Recv(u3, source=1, tag=(iteration+10))

        # Print Temperature distributions in each Iteration
        print(f'\n\n Iteration {iteration+1}: ')
        print(f'\n Temperature in Omega 1: \n {u1.reshape(4,4)} \n\n Temperature in Omega 2: \n {u2.reshape(7,4)} \n\n Temperature in Omega 3: \n {u3.reshape(4,4)}')


### Iteration Process without MPI

# print('\n Initial conditions: ')
# print(f'\n u_1: \n {u1.reshape(4,4)} \n\n u_2: \n {u2.reshape(7,4)} \n\n u_3: \n {u3.reshape(4,4)}')

# iterations = 10
# while iterations > 0:
#     Gamma1, Gamma2 = gamma(u1, u3)
#     u2_new = solve_omega2(Gamma1, Gamma2, A2)
#     u2_new = reset_walls_2(u2_new)
#     u2_new = relax(u2, u2_new)

#     Gamma1, Gamma2 = gamma(u2_new)
#     u1_new = solve_omega1(Gamma1, A1)
#     u3_new = solve_omega3(Gamma2, A1)
#     u1_new, u3_new = reset_walls_13(u1_new, u3_new)
#     u1_new = relax(u1, u1_new)
#     u3_new = relax(u3, u3_new)

#     u1 = u1_new
#     u2 = u2_new
#     u3 = u3_new

#     print(f'\n\n Iteration {11-iterations}: ')
#     print(f'\n u_1: \n {u1.reshape(4,4)} \n\n u_2: \n {u2.reshape(7,4)} \n\n u_3: \n {u3.reshape(4,4)}')
#     iterations -= 1
