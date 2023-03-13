import numpy as np
import random
from operators.core.route import Route


class Solution:
    def __init__(self, p, route_indexes:np.ndarray = None) -> None:
        """ 
        Parameters
        ----------
        Route_indexes : np.ndarray(dtype=int)
            a list of id for the routes
        """
        # If roue_indexes is not provided, build according to problem parameters. Else, use route_indexes
        route_indexes = generate_sub_arrays(p.nbr_patients, p.nbr_nurses) if route_indexes is None else route_indexes
        routes = []
        for route in route_indexes:
            patients = [p.patients[patient_id] for patient_id in route]
            route = Route(patients)
            routes.append(route)
        self.routes = routes
        self.matrix = route_indexes
        self.p = p

    def __len__(self):
        return len(self.routes)

    def __getitem__(self, index):
        return self.routes[index]
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.routes):
            raise StopIteration
        else:
            route = self.routes[self.current_index]
            self.current_index += 1
            return route
    def travel_time(self, p):
        t = 0
        for route in self:
            t += route.travel_time(p)
        return t
    def late_time(self, p):
        t = 0
        for route in self:
            t += route.late_time(p)
        return t
    def fitness(self, p, w: float = 1) -> float:
        """
        Returns the solution's fitness according to problem `p`, with weighted late_time by a factor of `w`.

        Parameters
        ----------
        p : Problem
            The problem at hand
        w : float
            The weight to five to late_time

        Returns
        -------
        fitness : float
            The fitness value of the given solution.
        """
        return self.travel_time(p) + w*self.late_time(p)

class Population:
    def __init__(self, pop_size, p, init="random", solutions: list=None):
        """Initializes a new population
        Parameters
        ----------
        pop_size: int
        p : Problem
            The problem at hand
        init : str
            The init mode, can be `random` or  `custom`
        solutions : list(Solutions)
            The list of solutions, eachb solution must be of type Solution
        """
        assert init in ["random", "custom"], "`init` must either be 'random', or 'custom'"
        if init == "custom" : assert solutions is not None, "Solutions `solutions`must be provided on custom init"
        if init == "random" : self.solutions = [Solution(p) for _ in range(pop_size)]
        elif init == "custom" : self.solutions = solutions

        self.pop_size = pop_size
        self.problem = p
            
    def __getitem__(self, index):
        return self.solutions[index]
    def __len__(self):
        return len(self.solutions)
    def __getitem__(self, index):
        return self.solutions[index]
    def __iter__(self):
        self.current_index = 0
        return self
    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            solution = self[self.current_index]
            self.current_index += 1
            return solution
    def sample(self, k):
        """
        Selects `k` random solutions taken from the population. Allows multi-picking of same solution.
        Parameters
        ----------
        k : int
            The number of random solutions to select.
        Returns
        -------
        solutions: list(Solutions)
            `k` randomly selected solutions from the population.
        """
        return random.sample(self.solutions, k)
    def get_best_time_population(self):
        return min(self, key=lambda x: x.travel_time(self.problem))
    
def fitness_from_matrix(solution_matrix, problem):
    solution = Solution(p = problem, route_indexes=solution_matrix)
    return solution.fitness(problem)

def generate_sub_arrays(N,n, permute="True"):
    arr = np.arange(1, N+1)
    arr = np.random.permutation(arr) if permute else arr

    indices = np.random.choice(np.arange(2, N+1), size=n-1, replace=False)
    indices.sort()

    sub_arrays = np.split(arr, indices)
    return np.array(sub_arrays, dtype=object)