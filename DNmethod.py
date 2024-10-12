from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import spsolve as solve

class DN_Method():

    def __init__(self, h, t_n=15, t_h=40, t_w=5, t_r=20, omega=0.8):
        self.h = h
        self.t_n = t_n
        self.t_h = t_h
        self.t_w = t_w
        self.t_r = t_r
        self.omega = omega
        self.n = (1 / h) + 1
        # Initialize MPI communication
        self.comm = MPI.Comm.Clone( MPI.COMM_WORLD )
        self.rank = self.comm.Get_rank()
    
    # Initialize temperature vectors
    def init_u(self):
        u1_init = np.zeros(int(self.n ** 2))
        u2_init = np.zeros(int(self.n * ((2 * self.n) - 1)))
        u3_init = np.zeros(int(self.n ** 2))

        # Fill u1_init
        for ix in range(len(u1_init)):
            i = ix % self.n
            j = ix // self.n
            if (j == 0 or j == self.n - 1) and i != 0:
                u1_init[ix] = self.t_n
            elif i == 0:
                u1_init[ix] = self.t_h
            else:
                u1_init[ix] = self.t_r
    
        # Fill u2_init
        for ix in range(len(u2_init)):
            i = ix % self.n
            j = ix // self.n
            if j == 0 and i != 0:
                u2_init[ix] = self.t_w
            elif i == self.n - 1 and j != 0 and j < self.n:
                u2_init[ix] = self.t_n
            elif i == 0 and j != 2 * self.n - 2 and j >= self.n - 1:
                u2_init[ix] = self.t_n
            elif j == 2 * self.n - 2 and i != self.n - 1:
                u2_init[ix] = self.t_h
            elif (j == 0 and i == 0) or (j == 2 * self.n - 2 and i == self.n - 1):
                u2_init[ix] = self.t_n
            else:
                u2_init[ix] = self.t_r
        
        # Fill u3_init
        for ix in range(len(u3_init)):
            i = ix % self.n
            j = ix // self.n
            if (j == 0 or j == self.n - 1) and i != 0:
                u3_init[ix] = self.t_n
            elif i == 0:
                u3_init[ix] = self.t_h
            else:
                u3_init[ix] = self.t_r

        return u1_init, u2_init, u3_init
    
    def gamma(self, vec1, vec2=None):
        '''
        Derives boundary values (Gamma) from temperature vectors of the rooms.
        If two temperature vectors are provided, it extracts boundary values for small rooms (room 1 and room 3).
        If only one temperature vector is provided, it extracts boundary values for the large room (room 2).
        
        Parameters:
        vec1 (ndarray): Temperature vector of room 1 or 2.
        vec2 (ndarray, optional): Temperature vector of room 3.
        
        Returns:
        Gamma1 (ndarray): Boundary temperatures from the left interface.
        Gamma2 (ndarray): Boundary temperatures from the right interface.
        '''
        Gamma1, Gamma2 = np.zeros(int(self.n)), np.zeros(int(self.n))
        max_ix = self.n - 1
        for ix in range(len(Gamma1)):
            if vec2 is not None: # Two temperature vectors have been provided
                Gamma1[ix] = vec1[int(max_ix + ix * self.n)]
                Gamma2[ix] = vec2[int(max_ix + (max_ix - ix) * self.n)]
            else: # For room 2
                Gamma1[ix] = vec1[int(ix * self.n)]  
                Gamma2[ix] = vec1[int((self.n * self.n - 1) + ix * self.n)] 
        return Gamma1, Gamma2
    
    def set_uDC(self, Gamma1, Gamma2):
        u_DC = np.zeros(int(self.n * ((2 * self.n) - 1)))

        for ix in range(len(u_DC)):
            i = ix % self.n
            j = ix // self.n

            # Dirichlet Condition for Interfaces Gamma 1, and Gamma 2
            if i == 1 and j < self.n:
                u_DC[ix] = Gamma1[int(j)]
            elif i == self.n - 2 and j >= self.n - 1:
                u_DC[ix] = Gamma2[int(j - self.n - 1)]
            
            # Dirichlet Condition for walls
            elif i == 0 and j == 1:
                u_DC[ix] = self.t_n
            elif j == 1 and i < self.n:
                u_DC[ix] = self.t_w
            elif (i == self.n - 1 and j < self.n - 1) or (i == 0 and j == self.n - 2) or (i == self.n - 1 and j == self.n) or (i == self.n - 1 and j == self.n * 2 - 3):
                u_DC[ix] = self.t_n
            elif i > 0 and j == self.n * 2 - 3:
                u_DC[ix] = self.t_h
            elif i == 1 and j >= self.n:
                u_DC[ix] = self.t_n

        u_DC = (- 1 / (self.h ** 2)) * u_DC
        return u_DC
    
    def set_uNC(self, Gamma):
        u_NC = np.zeros(int(self.n ** 2))

        for ix in range(len(u_NC)):
            i = ix % self.n
            j = ix // self.n

            # Neumann Condition for Interface Gamma 1 or Gamma 2
            if i == self.n - 1:
                u_NC[ix] = - 1 / self.h * Gamma[int(j)]

            # Dirichlet Condition for walls
            elif i == 1 and j > 0 and j < self.n - 1:
                u_NC[ix] = (- 1 / (self.h ** 2)) * self.t_h
            elif i > 1 and (j == 1 or j == self.n - 2):
                u_NC[ix] = (- 1 / (self.h ** 2)) * self.t_n
        
        return u_NC

    # Step 1: Solve problem for room 2
    def solve_omega2(self, Gamma1, Gamma2, A):
        '''
        Solves for the temperature distribution in room 2 given the boundary values (Gamma1 and Gamma2).

        Parameters:
        Gamma1 (ndarray): Boundary values from room 1.
        Gamma2 (ndarray): Boundary values from room 3.
        A (ndarray): Coefficient matrix for room 2.

        Returns:
        u2_new (ndarray): New temperature distribution in room 2.
        '''
        u_DC = self.set_uDC(Gamma1, Gamma2)
        u2_new = solve(A, u_DC)
        return u2_new
    
    # Step 2: Solve problem for room 1 and 3 (single functions for each room)
    def solve_omega1(self, Gamma1, A): 
        '''
        Solves for the temperature distribution in room 1 given the boundary values from room 2.

        Parameters:
        Gamma1 (ndarray): Boundary values from room 2.
        A (ndarray): Coefficient matrix for room 1.

        Returns:
        u1_new (ndarray): New temperature distribution in room 1.
        '''
        u_NC = self.set_uNC(Gamma1)
        u1_new = solve(A, u_NC)
        return u1_new

    def solve_omega3(self, Gamma2, A):
        '''
        Solves for the temperature distribution in room 3 given the boundary values from room 2.

        Parameters:
        Gamma2 (ndarray): Boundary values from room 2.
        A (ndarray): Coefficient matrix for room 3.

        Returns:
        u3_new (ndarray): New temperature distribution in room 3.
        '''
        Gamma2_reversed = Gamma2[::-1]
        u_NC = self.set_uNC(Gamma2_reversed)
        u3_new = solve(A, u_NC)
        return u3_new
    
    # Between Step 2 and Step 3: Reset wall temperatures in new temperature vectors
    def reset_walls_13(self, u1, u3): 
        '''
        Resets the wall temperatures to predefined values after solving for the temperature distribution in rooms 1 and 3.

        Parameters:
        u1 (ndarray): Temperature vector for room 1.
        u3 (ndarray): Temperature vector for room 3.

        Returns:
        tuple: Updated temperature vectors for room 1 and room 3 with boundary (wall) values reset.
        '''
        u1_walls = np.zeros(int(self.n ** 2))
        u3_walls = np.zeros(int(self.n ** 2))

        # Fill new vectors with wall temperatures and new room temperatures
        for ix in range(len(u1_walls)):
            i = ix % self.n
            j = ix // self.n
            if (j == 0 or j == self.n - 1) and i != 0:
                u1_walls[ix] = self.t_n
                u3_walls[ix] = self.t_n
            elif i == 0:
                u1_walls[ix] = self.t_h
                u3_walls[ix] = self.t_h
            else:
                u1_walls[ix] = u1[ix]
                u3_walls[ix] = u3[ix]

        return u1_walls, u3_walls

    def reset_walls_2(self, u2):  
        '''
        Resets the wall temperatures to predefined values after solving for the temperature distribution in room 2.

        Parameters:
        u2 (ndarray): Temperature vector for room 2.

        Returns:
        ndarray: Updated temperature vector for room 2 with boundary (wall) values reset.
        '''
        u2_walls = np.zeros(int(self.n * ((2 * self.n) - 1)))
        
        # Fill new vector with wall temperatures and new room temperatures
        for ix in range(len(u2_walls)):
            i = ix % self.n
            j = ix // self.n
            if j == 0 and i != 0:
                u2_walls[ix] = self.t_w
            elif i == self.n - 1 and j != 0 and j < self.n:
                u2_walls[ix] = self.t_n
            elif i == 0 and j != 2 * self.n - 2 and j >= self.n - 1:
                u2_walls[ix] = self.t_n
            elif j == 2 * self.n - 2 and i != self.n - 1:
                u2_walls[ix] = self.t_h
            elif (j == 0 and i == 0) or (j == 2 * self.n - 2 and i == self.n - 1):
                u2_walls[ix] = self.t_n
            else:
                u2_walls[ix] = u2[ix]

        return u2_walls

    # Step 3: Relaxation
    def relax(self, u, u_new):
        '''
        Applies relaxation method to smooth the temperature values by combining the old and new solutions.

        Parameters:
        u (ndarray): Current temperature vector.
        u_new (ndarray): New calculated temperature vector.

        Returns:
        ndarray: Relaxed temperature vector.
        '''
        u_new = self.omega * u_new + (1 - self.omega) * u
        return u_new
    
    def dn_iteration(self, A1, A2, iterations=10):
        u1, u2, u3 = self.init_u()

        # Print initial conditions in process 2
        if self.rank == 2:
            print(f'\n Initial Conditions:')
            print(f'\n Temperature in Omega 1: \n {u1.reshape(int(self.n), int(self.n))} \n\n Temperature in Omega 2: \n {u2.reshape(int(2 * self.n - 1),int(self.n))} \n\n Temperature in Omega 3: \n {u3.reshape(int(self.n),int(self.n))}')
        
        for iteration in range(iterations):
    
            # Process 0 computes the temperature distribution for room 2 (Omega 2) in each iteration
            if self.rank == 0:
                # Receive Gamma1 and Gamma2 from process 1
                Gamma1 = np.empty(int(self.n))
                Gamma2 = np.empty(int(self.n))
                self.comm.Recv(Gamma1, source=1, tag=iteration)
                self.comm.Recv(Gamma2, source=1, tag=(iteration+10))

                # Compute u2_new
                u2_new = self.solve_omega2(Gamma1, Gamma2, A2)
                u2_new = self.reset_walls_2(u2_new)

                # Relaxation
                u2_new = self.relax(u2, u2_new)

                # Send Gamma1 and Gamma2 to process 1
                Gamma1, Gamma2 = self.gamma(u2_new)
                self.comm.Send(Gamma1, dest=1, tag=iteration)
                self.comm.Send(Gamma2, dest=1, tag=(iteration+10))

                # Send u2_new to process two
                self.comm.Send(u2_new, dest=2, tag=iteration)

                # Update u2
                u2 = u2_new  

            # Process 1 computes the temperature distributions for room 1 (Omega 1) and room 3 (Omega 3) in each iteration
            if self.rank == 1:
                # Send Gamma1 and Gamma2 to process 0
                Gamma1, Gamma2 = self.gamma(u1, u3)
                self.comm.Send(Gamma1, dest=0, tag=iteration)
                self.comm.Send(Gamma2, dest=0, tag=(iteration+10))

                # Receive Gamma1 and Gamma2 from process 0
                Gamma1 = np.empty(int(self.n))
                Gamma2 = np.empty(int(self.n))
                self.comm.Recv(Gamma1, source=0, tag=iteration)
                self.comm.Recv(Gamma2, source=0, tag=(iteration+10))

                # Compute u1_new and u3_new
                u1_new = self.solve_omega1(Gamma1, A1)
                u3_new = self.solve_omega3(Gamma2, A1)
                u1_new, u3_new = self.reset_walls_13(u1_new, u3_new)

                # Relaxation
                u1_new = self.relax(u1, u1_new)
                u3_new = self.relax(u3, u3_new)

                # Send u1_new and u3_new to process two
                self.comm.Send(u1_new, dest=2, tag=iteration)
                self.comm.Send(u3_new, dest=2, tag=(iteration+10))

                # Update u1 and u3
                u1 = u1_new
                u3 = u3_new

            # Process 2 prints the temperature distributions for each room in each iteration in form of matrices
            if self.rank == 2:
                # Receive new u1, u2 and u3 vectors from processes 0 and 1
                u1 = np.empty(int(self.n ** 2))
                u2 = np.empty(int(self.n * ((2 * self.n) - 1)))
                u3 = np.empty(int(self.n ** 2))   
                self.comm.Recv(u2, source=0, tag=iteration)
                self.comm.Recv(u1, source=1, tag=iteration)
                self.comm.Recv(u3, source=1, tag=(iteration+10))

                # Print Temperature distributions in each Iteration
                print(f'\n\n Iteration {iteration+1}: ')
                print(f'\n Temperature in Omega 1: \n {u1.reshape(int(self.n),int(self.n))} \n\n Temperature in Omega 2: \n {u2.reshape(int(2 * self.n - 1),int(self.n))} \n\n Temperature in Omega 3: \n {u3.reshape(int(self.n),int(self.n))}')
