import loader
import population

p = loader.Problem("data/train_0.json")
pop = population.Population(2, p)
eg_sol = pop[0]
eg_route = eg_sol[0]

print(pop[1].matrix)