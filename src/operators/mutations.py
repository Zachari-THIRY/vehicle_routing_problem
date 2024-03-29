import numpy as np
from operators.crossovers import smart_insert
from operators.core.loader import Problem
def mutate_solution(solution_matrix: np.ndarray, problem:Problem, parameters:dict, n_mutations:int = 3):
    """Does a crossover over a random selection of 2 routes from the solution matrix,
    and does `n_mutations` within random roads.

    Parameters
    ----------
    solution_matrix: np.ndarray()
        The matrix of the solution to be mutated
    n_mutations : int
        The number of mutations to apply
    parameters : the global_parameters
    Returns
    -------
        new_route_indexes : list(list(ids))
            A new solution's route_indexes with effectuated random_crossover.
    """
    parameters = parameters["mutation_parameters"]
    mode = parameters["mode"]
    new_route_indexes = np.copy(solution_matrix)

    i,j = np.random.choice(len(new_route_indexes), 2)
    new_route_indexes[i], new_route_indexes[j] = intra_cross_over(new_route_indexes[i], new_route_indexes[j])

    if mode == "inverse":
        mut_idx = np.random.choice(len(new_route_indexes), n_mutations)
        c = 0
        for idx in mut_idx :
            new_route_indexes[idx] = mutate_route(new_route_indexes[idx], problem=problem, parameters=parameters)
            c += 1  
    
    elif mode == "appendix":
            new_route_indexes = appendix_mutation(parent = new_route_indexes, p = problem, fitness = parameters["fitness"])
    
    return new_route_indexes

def mutate_route(route_indexes:np.ndarray, problem:Problem, parameters:dict):
    """
    route_indexes : np.ndarray
        a 1D array of patient ids (definint a route)
    parameters : dict
        The mutation_parameters
    """
    mode = parameters["mode"]
    modes = ["inverse", "appendix"]
    assert mode in modes, "Mutation parameter `mode` must be in {}".format(modes)

    if mode == modes[0]:
        return inverse_mutation(route_indexes)
    elif mode == modes[1]:
        return appendix_mutation(parent = route_indexes, p = problem, fitness = parameters["fitness"])


def appendix_mutation(parent:np.ndarray,p, fitness:str):
    L = np.shape(parent)[0]
    pop_idx = np.random.randint(0, len(parent))
    pop_row = parent[pop_idx]
    for i in range(L):
        parent[i] = np.setdiff1d(parent[i], pop_row)
    
    for pid in pop_row:
        parent = smart_insert(parent, pid, p, fitness)
    return parent

def inverse_mutation(parent):
    """ Takes as input a route p1 and performs in-place inverse mutation
    Parameters
    ----------
    parent : Route.ids
        the parent route ids
    """
    L = len(parent)
    # If parent is too short, then don't inverse_mutate
    if L <= 1: 
        return parent
    
    # If length is sufficient, do a mutation
    else :
        i,j = np.random.choice(L, 2, replace=False)
        if i > j:
            i, j = j, i

        # Invert the segment between i and j
        parent[i:j+1] = parent[i:j+1][::-1]

        return parent

def intra_cross_over(p1,p2):
    """Does a crossover, within the routes of a same solution. Is called by intra_cross_over()

    Parameters
    ----------
        p1 : list(int)
            The list of ids of parent 1
        p2 : list(int)
            The list of ids of parent 2
    Returns
    -------
    list(ids),lsit(ids)
        The ids corresponding to the routes of child 1 and child 2
        
    """

    l1, l2 = len(p1), len(p2)

    if l1 == 0 or l2 == 0 : return p1,p2 # If one route is empty, return the parents (crossover impossible)

    c1 = np.empty(l1, dtype=int)
    c2 = np.empty(l2, dtype=int)
    if not l1 > l2 : 
        p1,p2 = p2,p1 # Making sure p1 is the longer route
        c1,c2 = c2, c1
    min_length = min(l1, l2)
    x_over_point = np.random.randint(0,min_length)
    c1[:x_over_point] = p2[:x_over_point]
    c1[x_over_point:] = p1[x_over_point:]
    c2[:x_over_point] = p1[:x_over_point]
    c2[x_over_point:] = p2[x_over_point:]

    return c1, c2