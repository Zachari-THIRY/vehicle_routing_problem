from operators.core.population import Population

def parent_selection(population, p, mode, n_parents:int=None):
    """
    Returns a population issued from parent_tournament

    Parameters
    ----------
    population: Population
        The population from which to sample parents.
    p: Problem
        The associated problem.

    Returns
    -------
    new_pop: Population
        Containing the selected parents from tournement_selection.
    """
    modes = ["tournament", "full_elitism"]
    assert mode == "tournament", f"Provided mode isn't implemented. Try mode in {modes}"

    n_parents = len(population) if n_parents is None else n_parents
    pop_size = len(population)
    parents = []
    for i in range(pop_size):
        parent = tournament_round(population, problem=p, k_opponents=3)
        parents.append(parent)
    return Population(pop_size, p, init="custom", solutions=parents)

def tournament_round(population,problem, k_opponents=3, w_fitness = 1):
    """
    Parameters
    ----------
    population : Population
        The initial population, of type Population
    problem : Problem
        The problem at hand
    k_opponents : int
        Number of opponents on which to perform the tournament.
    w_fitness : float
        The fitness coefficient for travel_time ( travel_time + w*late_time)
 
    Returns
    -------
    winner : Solution
        The winner of the parent tournament
    """
    tournament = population.sample(k_opponents)
    winner = min(tournament, key=lambda x: x.fitness(problem, w_fitness)) # Calculate fitness using fitness_func
    return winner

def full_elitism(population, p, n_parents:int):
    sorted_sols = sorted([sol for sol in population], key = lambda x: x.fitness(p))[0:n_parents]
    return Population(n_parents, p=p, solutions=sorted_sols )
