from mpi4py import MPI
import numpy as np
from scipy.linalg import solve


# Initialize MPI communication
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
rank = comm.Get_rank()

# Size of steps (delta x) and relaxation parameter
h = 1/3
omega = 0.8

# Define initial temperature vectors
# Assume inner room temperature is 20°C
u1_init = np.array([32.5, 15, 15, 10, 40, 20, 20, 20, 40, 20, 20, 20, 32.5, 15, 15, 15])
u2_init = np.array([10, 5, 5, 10, 20, 20, 20, 15, 20, 20, 20, 15, 15, 20, 20, 15, 15, 20, 20, 20, 15, 20, 20, 20, 32.5, 40, 40, 32.5])
u3_init = np.array([32.5, 15, 15, 32.5, 40, 20, 20, 20, 40, 20, 20, 20, 32.5, 15, 15, 15])

# Coefficient matrix A for rooms 1 and 3
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

# Coefficient matrix A for room 2
A2 = 1/(h**2) * np.array([[-4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4]])


def gamma(vec1, vec2=None):
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
def solve_omega2(u1, u3, A):
    Gamma1, Gamma2 = gamma(u1, u3)
    u_DC = - 1/(h**2) * np.array([0, Gamma1[0], 0, 0, 0, Gamma1[1], 0, 0, 0, Gamma1[2], 0, 0, 0, Gamma1[3], Gamma2[0], 0, 0, 0, Gamma2[1], 0, 0, 0, Gamma2[2], 0, 0, 0, Gamma2[3], 0])
    u2_new = solve(A, u_DC)
    return u2_new

# Step 2: Solve problem for room 1 and 3 (single functions for each room)
def solve_omega1(u2, A):
    Gamma1, Gamma2 = gamma(u2)
    u_NC = - 1/h * np.array([0, 0, 0, Gamma1[0], 0, 0, 0, Gamma1[1], 0, 0, 0, Gamma1[2], 0, 0, 0, Gamma1[3]])
    u1_new = solve(A, u_NC)
    return u1_new

def solve_omega3(u2, A):
    Gamma1, Gamma2 = gamma(u2)
    u_NC = - 1/h * np.array([0, 0, 0, Gamma2[3], 0, 0, 0, Gamma2[2], 0, 0, 0, Gamma2[1], 0, 0, 0, Gamma2[0]])
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

def reset_walls(u1, u2, u3):
    u1_walls = np.array([32.5, 15, 15, 10, 40, u1[5], u1[6], u1[7], 40, u1[9], u1[10], u1[11], 32.5, 15, 15, 15])
    u2_walls = np.array([10, 5, 5, 10, u2[4], u2[5], u2[6], 15, u2[8], u2[9], u2[10], 15, 15, u2[13], u2[14], 15, 15, u2[17], u2[18], u2[19], 15, u2[21], u2[22], u2[23], 32.5, 40, 40, 32.5])
    u3_walls = np.array([32.5, 15, 15, 32.5, 40, u3[5], u3[6], u3[7], 40, u3[9], u3[10], u3[11], 32.5, 15, 15, 15])
    return u1_walls, u2_walls, u3_walls

#------------
# Iteration
#------------

u1 = u1_init
u2 = u2_init
u3 = u3_init

print('\n Initial conditions: ')
print(f'\n u_1: \n {u1.reshape(4,4)} \n\n u_2: \n {u2.reshape(7,4)} \n\n u_3: \n {u3.reshape(4,4)}')

# iterations = 10
# while iterations > 0:
#     u2_new = solve_omega2(u1, u3, A2)
#     u1_new = solve_omega1(u2_new, A1)
#     u3_new = solve_omega3(u2_new, A1)
#     u1 = relax(u1, u1_new)
#     u2 = relax(u2, u2_new)
#     u3 = relax(u3, u3_new)
#     u1, u2, u3 = reset_walls(u1, u2, u3)
#     print(f'\n\n Iteration {11-iterations}: ')
#     print(f'\n u_1: \n {u1.reshape(4,4)} \n\n u_2: \n {u2.reshape(7,4)} \n\n u_3: \n {u3.reshape(4,4)}')
#     iterations -= 1

# Iteration process with MPI communication
num_iterations = 10

for iteration in range(num_iterations):
    if rank == 0:
        # Receive u1 and u3 (old values)
        u1 = np.empty(16)
        u3 = np.empty(16)
        comm.Recv(u1, source=1, tag=(iteration))
        comm.Recv(u3, source=1, tag=(iteration+10))

        # Compute u2_new
        u2_new = solve_omega2(u1, u3, A2)
        space1, u2_new, space3 = reset_walls(u1, u2_new, u3)

        # Relaxation
        u2_new = relax(u2, u2_new)

        # Send u2_new
        comm.Send(u2_new, dest=1, tag=iteration)

        # Update u2
        u2 = u2_new  

    if rank == 1:
        # Send u1 and u3 
        comm.Send(u1, dest=0, tag=iteration)
        comm.Send(u3, dest=0, tag=(iteration+10))

        # Receive u2_new
        u2_new = np.empty(28)
        comm.Recv(u2_new, source=0, tag=iteration)

        # Compute u1_new and u3_new
        u1_new = solve_omega1(u2_new, A1)
        u3_new = solve_omega3(u2_new, A1)
        u1_new, space2, u3_new = reset_walls(u1_new, u2_new, u3_new)

        # Relaxation
        u1_new = relax(u1, u1_new)
        u3_new = relax(u3, u3_new)

        print(f'\n\n Iteration {iteration+1}: ')
        print(f'\n u_1: \n {u1_new.reshape(4,4)} \n\n u_2: \n {u2_new.reshape(7,4)} \n\n u_3: \n {u3_new.reshape(4,4)}')

        # Update u1 and u3
        u1 = u1_new
        u3 = u3_new

