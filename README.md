# project3NumericalAlg

**Authors:** Johanna Rissbacher, Jule Grimm

**Note:** We both worked on all parts of the project together

## Instructions

The main.py file has to be executed using three processes. 
This can be done by using the command <mpiexec /np 3 python main.py> in the command line.

## Project Structure

The project contains the following files:

- **DNmethod.py**: Contains classes to run Dirichlet Neumann iteration on different apartments
- **sparse_matrix.py**: Contains functions to create sparse matrices for coefficient matrices for different rooms
- **main.py**: Executes Dirichlet Neumann iteration for different tasks
- **apartment_21.png**: Plot of apartment layout 1 with n = 21
- **apartment2_101.png**: Plot of apartment layout 2 with n = 101
- **Matrices.txt**: Temperature matrices after 10th iteration for each room with apartment layout 1 and n = 4

## Class Structure (DNmethod.py)

- **Apartment**
    - **DN_Method**
        - **DN_Method4Rooms**

## Class Descriptions

### Apartment
- **Inputs**: 
    - \delta x (h): mesh width
    Optional Arguments:
    - t_n: normal wall temperature
    - t_h: Wall with heating
    - t_w: Wall with window
    - t_r: Initial interior room temperature
    - omega: relaxation parameter
- **Methods**:
    - `init_u`: Initializes temperature vectors for the three rooms in the apartment. 
    - `gamma`: Computes boundary temperatures (Gamma) at the interfaces 
    - `set_uDC`: Sets the Dirichlet conditions for the temperature vector of room 2
    - `set_uNC`: Sets the Neumann conditions for the temperature vector of rooms 1 and 3

### DN_Method (inherits from Apartment)
- **Methods**: 
    - `solve_omega2`: Solves for the temperature distribution in room 2
    - `solve_omega1`: Solves for the temperature distribution in room 1
    - `solve_omega3`: Solves for the temperature distribution in room 3
    - `reset_walls_13`: Resets the wall temperatures to predefined boundary values in the temperature vectors of rooms 1 and 3
    - `reset_walls_2`: Resets the wall temperatures to predefined boundary values in the temperature vectors of room 2
    - `relax`: Applies relaxation method to smooth the temperature values by combining the old and new solutions
    - `dn_iteration`: Executes Dirichlet Neumann iteration in three processes

## DN_Method4Rooms (inherits from DN_Method)
- **Purpose**: The methods that were defined before for the DN iteration are updated and added with other methods for the 4 room apartment layout
- **Methods**:
    - `init_u`: Initializes temperature vectors for the four rooms in the apartment. 
    - `gamma123`: Computes boundary temperatures (Gamma) at interfaces 123 for room 2
    - `gamma3`: Computes boundary temperatures (Gamma) at the interface 3 for room 4
    - `set_uDC`: Sets the Dirichlet conditions for the temperature vector of room 2
    - `set_uNC_room4`: Sets the Neumann conditions for the temperature vector of room 4
    - `solve_omega2`: Solves for the temperature distribution in room 2
    - `solve_omega4`: Solves for the temperature distribution in room 4
    - `reset_walls_2`: Resets the wall temperatures to predefined boundary values in the temperature vectors of room 2
    - `reset_walls_4`: Resets the wall temperatures to predefined boundary values in the temperature vectors of room 4
    - `dn_iteration`: Executes Dirichlet Neumann iteration for four rooms in three processes