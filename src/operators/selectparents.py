from operators.core.population import Population, Solution
from operators.core.loader import Problem

def parent_selection(population, p, mode, parameters:int=None):
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
    modes = ["tournament", "full_elitism", "n_elitism + tournaments"]
    assert mode == "tournament", f"Provided mode isn't implemented. Try mode in {modes}"
    pop_size = len(population)

    if mode == "tournament":
        # Asserting some conditions
        n_parents = parameters
        condition = (type(n_parents) == int and n_parents >= 0 and n_parents % 2 == 0) or parameters == None
        assert condition , f"in mode '{mode}', parameters must be a positive even integer."

        # Filling the parents list
        # TODO : replace this with numpy array for faster execution
        n_parents = pop_size if n_parents is None else n_parents
        parents = []
        for _ in range(pop_size):
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
        The winner of the parent tournament round
    """
    tournament = population.sample(k_opponents)
    winner = min(tournament, key=lambda x: x.fitness(problem, w_fitness)) # Calculate fitness using fitness_func
    return winner

def n_elitism(population : Population, p: Problem, n_parents:int) -> list:
    """
    Returns
    -------
    sorted_solutions : list(Solution)
        The n_parent best solutions in the population, sorted from best to worst
    """
    assert n_parents <= len(population)
    sorted_sols = sorted([sol for sol in population], key = lambda x: x.fitness(p))[0:n_parents]
    return sorted_sols
