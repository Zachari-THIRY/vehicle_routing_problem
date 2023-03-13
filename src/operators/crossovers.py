import numpy as np
from operators.core.route import get_route_from_ids
from operators.core.loader import Problem

def crossovers(solutions:list,problem:Problem, parameters:dict):
    """
    Applies crossover over a whole list of solutions.

    Parameters
    ----------
    solutions: list
        A list of solution matrixes
    Returns
    -------
    xov_solution_matrixes : list(np.ndarray 2D)
        The solution matrixes after applied crossovers
    """
    xov_solutions_matrixes = []
    for i in range(len(solutions)//2):
        children = crossover(p1 = solutions[i].matrix ,p2 = solutions[i+1].matrix, problem=problem, parameters=parameters)
        for child in children : 
            xov_solutions_matrixes.append(child)
    return xov_solutions_matrixes

def crossover(p1,p2, problem ,parameters):

    available_modes = ["appendix", "extra"]

    parameters = parameters["crossover_parameters"]
    mode = parameters["mode"]
    fitness = parameters["fitness"]
    assert mode in available_modes, "Parameter mode bust be in {AVAILABLE_MODES} but {MODE} was given.".format(AVAILABLE_MODES = available_modes, MODE = mode)

    if mode == available_modes[0]:
        return appendix_cross_over(p1,p2, problem, fitness)
    if mode == available_modes[1]:
        return extra_cross_over(p1,p2)

def extra_cross_over(p1, p2):
    """
    Selects a random splitting point (same for each parent), keeps the first part, 
    and then completes if possible with elements from the other parent.
    Parameters
    ----------
    p1 : np.ndarray
        The matrix of parent solution 1
    p2 : np.ndarray
        The matrix of parent solution 2
    """
  

    # Gathering the ids in order and lengths

    p1_ids = np.concatenate([np.fromiter(subarr, dtype=int) for subarr in p1]).tolist()
    p2_ids = np.concatenate([np.fromiter(subarr, dtype=int) for subarr in p2]).tolist()

    p1_lengths = [len(row) for row in p1]
    p2_lengths = [len(row) for row in p2]

    # Finding crossover point
    x_over_point = np.random.randint(1, len(p1_ids))

    # Splitting
    p1_head = p1_ids[:x_over_point]
    p1_tail = p1_ids[x_over_point:]
    p2_head = p2_ids[:x_over_point]
    p2_tail = p2_ids[x_over_point:]

    c1_ids = np.concatenate((p1_head, p2_tail), axis=0)
    c2_ids = np.concatenate((p2_head, p1_tail), axis=0)

    c1 = np.empty(len(p1_lengths), dtype=object)
    c2 = np.empty(len(p2_lengths), dtype=object)

    i1, i2 = 0,0
    for i,l in enumerate(p1_lengths):
        new_row =  c1_ids[i1:i1+l]
        c1[i] = new_row
        i1 += l
    for i,l in enumerate(p2_lengths):
        new_row =  c2_ids[i1:i1+l]
        c2[i] = new_row
        i2 += l    
    
    return c1, c2

def appendix_cross_over(p1,p2, problem:Problem, fitness:str):
    """
    Parameters
    ----------
    p1 : np.ndarray
        The matrix of parent 1
    p2 : np.ndarray
        The matrix of parent 2
    problem : Problem
        The problem at hand
    fitness : str
        The specified fitness : "fitness", "travel_time" or "late_time"

    Returns
    -------

    c1, c2 : (np.ndarray, np.ndarray) 
        The offspring crossovers as Solution.matrix
    """
    assert len(p1) == len(p2), "Matrixes `p1` and `p2` should have the same number of rows"
    L = len(p1)
    pop = np.random.randint(1, L, 2)
    
    to_pop_pids = p1[pop[0]], p2[pop[1]] # Contains the routes to be popped

    # Popping the routes in the other solution:
    for index in range(L) :
        p1[index] = np.setdiff1d(p1[index], to_pop_pids[1])
        p2[index] = np.setdiff1d(p2[index], to_pop_pids[0])

    # Re-inserting patients into the routes : 
    np.random.shuffle(to_pop_pids[0])
    np.random.shuffle(to_pop_pids[1])

    for pid in to_pop_pids[1]:
        p1 = smart_insert(p1, pid, problem, fitness)
    for pid in to_pop_pids[0]:
        p2 = smart_insert(p2, pid, problem, fitness)

    return p1, p2                

def smart_insert(parent, pid, p, fitness:str) -> np.ndarray:
    """
    Insert the patient pid in the best possible place inside parent
    Parameters
    ----------
    parent : np.ndarray
        Matrix representation of parent.
    pid : int
        The patient id.
    p : Problem
        The problem at hand.
    fitness : str
        With values in ["fitness", "travel_time", "late_time"], specifies which fitness to use to insert the rows.
    """
    lengths = np.array([len(row) for row in parent])
    lengths += 1

    # Build insertion penalties matrix

    insert_penalties = []
    for row in parent:
        row_penalties = []
        time_baseline = get_route_from_ids(row, p).travel_time(p)
        late_baseline = get_route_from_ids(row, p).late_time(p)
        for i in range(len(row) + 1):
            penalty = 0
            new_row = dumb_insert(row, pid, index = i)
            new_route = get_route_from_ids(new_row, p)
            if fitness in ["fitness", "travel_time"]:
                penalty += new_route.travel_time(p) - time_baseline
            if fitness in ["fitness", "late_time"]:
                penalty += new_route.late_time(p) - late_baseline
            row_penalties.append(penalty)
        insert_penalties.append(row_penalties)

    # Retrieve best position
    _, min_index = min(
    (val, (i, j))
    for i, row in enumerate(insert_penalties)
    for j, val in enumerate(row)
    )
    parent[min_index[0]] = dumb_insert(row = parent[min_index[0]], pid = pid, index = min_index[1])
    return parent

def dumb_insert(row, pid:int, index:int) -> np.ndarray:
    """Inserts a pid in the specified index
    Parameters
    ----------
    row : np.ndarray(int)
        The ids of the row
    pid : int
        The pid of the patient to insert
    index : int
        The position in the row at which to insert
    """
    new_row = np.insert(row, index, pid, axis=0)
    return new_row