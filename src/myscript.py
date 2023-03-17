# import sys
 
# sys.path.append("./operators/core")
# sys.path.append("./operators")

import operators.core.loader as loader
import operators.core.population as population
from operators.core.utils import display_round_info, display_solution, round_robin_init
import time
import GA

parameters = {
    "parent_selection" : {
        "mode" : "tournament",
        "parameters" : 16,
        ### Parameters for each mode ###
        "full_elitism" : {
            "n_elites": 16,
            "fitness": "late_time",     # can be in ["fitness", "travel_time", "late_time"]
            "w_fitness": 1
        },
        "tournament" : {
            "fitness": "fitness",   # can be in ["fitness", "travel_time", "late_time"]
            "w_fitness" : 1,
            "k_opponents" : 2,
            "n_parents": 100
        },
        "n_elitism+k_tournaments":{
            "fitness": "travel_time",
            "w_fitness": 1,
            "n_elites": 3,          # The number of parents in the elite section
            "n_parents": 27,        # The number of parents in the tournament section
            "k_opponents": 2,
        }
        ### End of modes ###
    },
    "crossover_parameters" : {
        "mode" : "appendix",        # can be in ["appendix", "extra"]
        "fitness" : "travel_time"       # can be in ["fitness", "travel_time", "late_time"]
    },
    "mutation_parameters": {
        "mode": "None",         # can be in ["appendix", "inverse", "None"]
        "fitness" : "travel_time"       # can be in ["fitness", "travel_time", "late_time"]
    },
    "print_period" : 10,
    "mixed_init": False
}

p = loader.Problem("data/train_0.json")
pop = population.Population(30, p)

# Mixed initialisation
round_robin_matrixes = round_robin_init(p)
rob_sol = population.Solution(p=p, route_indexes=round_robin_matrixes)
solutions = [rob_sol]*16

if parameters["mixed_init"]:
    for _ in range(10):
        # solutions.append(population.Solution(p=p))
        pass
    pop = population.Population(
        pop_size=16,
        p = p,
        init="custom",
        solutions=solutions
    )
### Running the GA ###

start = time.time()
for i in range(10):
    pop = GA.mutate_population(pop, p, parameters=parameters)
    if i % parameters["print_period"] == 0 : 
        display_round_info(pop, p, i)

display_round_info(population = pop, problem = p, epoch="end")

elapsed_time = time.time() - start
print("Elapsed time:\t", elapsed_time)

### End of the GA ###

### Saving solutions ###

display_solution(pop.get_best_time_population(), p, "none")
display_solution(pop[0], p, "none")