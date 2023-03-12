import numpy as np

def get_distance(d, p1, p2):
    """
    Returns the travel_time between p1 and p2. If p1 or p2 aren't specified, then the depot is considered.

    Parameters
    ----------
    p1: Patient
        The first patient
    p2: Patient
        The Second patient

    Returns
    -------
    t : int
        The travel time between p1 and p2
    """

    pid1 = 0 if p1 == None else p1.id
    pid2 = 0 if p2 == None else p2.id
    return d[pid1][pid2]

def tournament_selection(population,p, k=3, w = 1):
    """
    Returns
    -------
    winner : Solution
        The winner of the parent tournament
    """
    tournament = population.sample(k)
    winner = min(tournament, key=lambda x: x.fitness(p, w)) # Calculate fitness using fitness_func
    return winner

def generate_sub_arrays(N,n, permute="True"):
    arr = np.arange(1, N+1)
    arr = np.random.permutation(arr) if permute else arr

    indices = np.random.choice(np.arange(2, N+1), size=n-1, replace=False)
    indices.sort()

    sub_arrays = np.split(arr, indices)
    return np.array(sub_arrays, dtype=object)