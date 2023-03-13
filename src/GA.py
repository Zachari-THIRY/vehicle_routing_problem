from operators.core.population import Population, Solution
from operators.selectparents import parent_selection
from operators.core.loader import Problem
from operators.mutations import mutate_solution
from operators.crossovers import appendix_cross_over

def mutate_population(population:Population,problem:Problem, parameters: dict):
    """Takes as input a population, selects parents, applies crossver between different solutions, and then mutates each solution.
    Returns
    -------
    pop : Population
        The population resulting from a sequence of parent_selection, appendix_crossover, intra_crossover and mutation
    """
    assert len(population) %2 == 0, "Population length must be even"

    old_solutions = population.solutions                          # Type Solution
    new_solutions = parent_selection(population=population, p=problem, parameters=parameters)    # Type Solution, gets parents from tournament selection
   
    xov_solutions_matrixes = [] # List of Solution.matrix

    # Crossover between parents
    for i in range(len(new_solutions)//2):
        children = appendix_cross_over(p1 = new_solutions[i].matrix ,p2 = new_solutions[i+1].matrix, problem=problem)
        for child in children : 
            xov_solutions_matrixes.append(child)

    # Mutate within solutions
    mutated_solutions_idx = []
    for new_solution in xov_solutions_matrixes :
        new_solution_idx = mutate_solution(new_solution)
        mutated_solutions_idx.append(new_solution_idx)

    # Transform list of route_indexes in list of solutions
    new_solutions = [Solution(problem, route_indexes) for route_indexes in mutated_solutions_idx]
    
    # Pure elitism : 
    sorted_solutions = sorted(new_solutions + old_solutions, key=lambda x: x.fitness(problem), reverse=False)[0:len(population)]
    return Population(population.pop_size, population.problem, init='custom', solutions = sorted_solutions)