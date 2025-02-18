import numpy as np

def particle_swarm(f, x_dim, bounds, iter_tot=100):
    """
    Particle Swarm Optimization (PSO) algorithm following Algorithm 16.1 (gbest PSO).
    
    Resource: https://towardsdatascience.com/what-the-hell-is-particle-swarm-optimization-pso-simplest-explanation-in-python-be296fc3b1ab/

    Resource: https://web2.qatar.cmu.edu/~gdicaro/15382/additional/CompIntelligence-Engelbrecht-ch16.pdf
    First 4 pages give sufficient detail, equations, etc. to code PSO
    
    Parameters:
    f : function
        Objective function to minimize.
    x_dim : int
        Dimensionality of the search space.
    bounds : np.ndarray
        Array with shape (x_dim, 2) specifying lower and upper bounds for each dimension.
    iter_tot : int, optional
        Total number of function evaluations (default is 100).
    
    Returns:
    tuple
        Best found position, best fitness value, team name, and names (for logging purposes).
    """
    # Step 1: Initialize swarm
    # 1a. Define the swarm size
    swarm_size = ...  # e.g., 10
    
    # 1b. Initialize particle positions randomly within bounds
    #    Hint: you can use np.random.uniform(...) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    positions = ...
    
    # 1c. Initialize velocities to zero (or some random initialization)
    velocities = ...
    
    # 1d. Set personal best positions to initial positions
    personal_best_positions = ...
    
    # 1e. Evaluate personal best fitness at the initial positions
    personal_best_fitness = ...
    
    # 1f. Determine global best position/fitness from initial personal bests
    global_best_idx = ...       # Index of best particle
    global_best_position = ...  # Best position found so far
    global_best_fitness = ...   # Best fitness value found so far
    
    # Step 2: Compute max_iterations based on evaluation budget
    max_iterations = iter_tot // swarm_size
    
    # Optimization loop
    for iter_num in range(max_iterations):
        # Step 3: Update inertia weight
        w = ...
        
        # Step 4: Generate random coefficients for velocity update
        r1, r2 = ...
        
        # Step 5: Update velocities using PSO velocity update equation
        #         velocities = (w * velocities 
        #                       + c1 * r1 * (personal_best_positions - positions) 
        #                       + c2 * r2 * (global_best_position - positions))
        c1= ...
        c2= ...
        velocities = ...
        
        # Step 6: Update positions
        positions += velocities
        
        # Enforce boundary constraints to keep particles within the search space
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])  

        # Step 7: Evaluate fitness at the new positions
        fitness = ...
        
        # Step 8: Update personal bests where improvement is found
        improved = ...                          # Boolean mask for improvements
        personal_best_positions[improved] = ... # Update positions
        personal_best_fitness[improved] = ...   # Update fitness values
        
        # Step 9: Update global best position (if any improvement)
        global_best_idx = ...       # Index of best personal best
        global_best_position = ...  # Update global best position
        global_best_fitness = ...   # Update global best fitness
    
    # Optional placeholders for logging (team name, student IDs, etc.)
    team_name = ['8']  
    names = ['01234567']
    
    # Return best solution found
    return global_best_position, global_best_fitness, team_name, names
