import numpy as np

def get_distance(d, p1, p2):
    """
    Parameters
    ----------
    p1: Patient
        The first patient
    p2: Patient
        The Second patient
    """
    return d[p1.id][p2.id]

def tournament_selection(population,p, k=3):
    tournament = population.sample(k)
    winner = min(tournament, key=lambda x: x.fitness(p)) # Calculate fitness using fitness_func
    return winner

def generate_sub_arrays(N,n):
    arr = np.arange(1, N+1)
    permuted_arr = np.random.permutation(arr)

    indices = np.random.choice(np.arange(2, N+1), size=n, replace=False)
    indices.sort()

    sub_arrays = np.split(permuted_arr, indices)
    return np.array(sub_arrays, dtype=object)