from operators.core.population import Population, Solution
from operators.core.loader import Problem

def parent_selection(population, p, parameters:dict):
    """
    Returns a population issued from parent_tournament

    Parameters
    ----------
    population: Population
        The population from which to sample parents.
    p: Problem
        The associated problem
    parameters: dict
        Must contain a "parent_selection" key.

    Returns
    -------
    new_pop: Population
        Containing the selected parents.
    """
    parameters = parameters["parent_selection"]
    modes = ["tournament", "full_elitism", "n_elitism+k_tournaments"]
    mode = parameters["mode"]
    parameters = parameters[mode]
    assert mode in modes[0:3], "Provided mode isn't implemented. Try mode in {}".format(modes)
    pop_size = len(population)

    if mode == modes[0]:
        # Asserting some conditions
        n_parents = parameters["n_parents"]
        condition = (type(n_parents) == int and n_parents >= 0 and n_parents % 2 == 0) or parameters == None
        assert condition , "In mode '{}', n_parents must be a positive even integer.".format(mode)

        # Filling the parents list
        # TODO : replace this with numpy array for faster execution
        n_parents = pop_size if n_parents is None else n_parents
        parents = []
        for _ in range(pop_size):
            parent = tournament_round(population, problem=p, parameters=parameters)
            parents.append(parent)
    if mode == modes[1]:
        # Asserting some conditions
        n_parents = parameters
        condition = (type(n_parents) == int and n_parents >= 0 and n_parents % 2 == 0) or parameters == None
        assert condition , "In mode '{}', parameters must be a positive even integer.".format(mode)
        assert n_parents <= len(population), "Can not sample more than population length in mode '{}'. Try parameters <= len(population)".format(mode)

        # Filling the parents
        parents = n_elitism(population=population, problem=p, parameters=parameters)

    if mode == modes[2]:
        # n_elitism + tournament
        # Asserting some conditions
        assert type(parameters) == tuple and len(parameters) == 2, "in mode {}, `parameters`must be a tuple of (n_elites, k_tournaments)".format(mode)
        n_elites, k_tournaments = parameters
        # Filling in the elites
        parents = n_elitism(population=population, problem=p, n_parents=n_elites)
        # Filling in the others issued from tournament
        for _ in range(k_tournaments):
            parent = tournament_round(population, problem=p, parameters=parameters)
            parents.append(parent)

    return Population(pop_size, p, init="custom", solutions=parents)

def tournament_round(population:Solution,problem:Problem, parameters:dict):
    """
    Parameters
    ----------
    population : Population
        The initial population, of type Population
    problem : Problem
        The problem at hand
    parameters : dict
        The dictionary of root parameters
 
    Returns
    -------
    winner : Solution
        The winner of the parent tournament round
    """
    # Gatheting parameters
    fitness_func = parameters["fitness"]
    k_opponents = parameters["k_opponents"]

    # Getting the right fitness function
    if fitness_func == "fitness":
        key_func = lambda x : x.fitness(problem, parameters["w_fitness"])
    if fitness_func == "travel_time":
        key_func = lambda x : x.travel_time(problem)
    if fitness_func == "late_time":
        key_func = lambda x : x.late_time(problem)

    tournament = population.sample(k_opponents)
    winner = min(tournament, key= key_func) # Calculate fitness using fitness_func
    return winner

def n_elitism(population : Population, problem: Problem, parameters:dict) -> list:
    """
    Returns
    -------
    sorted_solutions : list(Solution)
        The n_parent best solutions in the population, sorted from best to worst
    problem : Problem
        The problem at hand.
    paramters : dict
        The dictionary of root_parameters
    """
    n_parents = parameters["n_elites"]
    fitness_func = parameters["fitness"]

    assert n_parents <= len(population)

    # Getting the right fitness function
    if fitness_func == "fitness":
        key_func = lambda x : x.fitness(problem, parameters["w_fitness"])
    if fitness_func == "travel_time":
        key_func = lambda x : x.travel_time(problem)
    if fitness_func == "late_time":
        key_func = lambda x : x.late_time(problem)

    # Applying elitism
    sorted_sols = sorted([sol for sol in population], key = key_func)[0:n_parents]
    return sorted_sols