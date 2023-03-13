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
                        "mode" : "full_elitism",
                        "parameters" : 16,
                    },
    "crossover_parameters" : {
        "mode" : "appendix"
    }
}

p = loader.Problem("data/train_0.json")
pop = population.Population(16, p)

# Mixed initialisation
round_robin_matrixes = round_robin_init(p)
rob_sol = population.Solution(p=p, route_indexes=round_robin_matrixes)
solutions = [rob_sol]*6

for _ in range(10):
    solutions.append(population.Solution(p=p))
pop = population.Population(
    pop_size=16,
    p = p,
    init="custom",
    solutions=solutions
)
### Running the GA ###

start = time.time()
for i in range(100):
    pop = GA.mutate_population(pop, p, parameters=parameters)
    if i %50 == 0 : 
        display_round_info(pop, p, i)

display_round_info(population = pop, problem = p, epoch="end")

elapsed_time = time.time() - start
print("Elapsed time:\t", elapsed_time)

### End of the GA ###

### Saving solutions ###

display_solution(pop.get_best_time_population(), p, "none")