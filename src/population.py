import numpy as np
from utils import get_distance, generate_sub_arrays, tournament_selection
import random
from loader import Problem

class Route:
    """
    self.patients : python list of dtype Patient
    self.ids : np.array of ids (used for computation, not representation)
    """
    def __init__(self, patients) -> None:
        """
        Takes as input : patients, the arrat of dtype Patient
        """
        self.patients = patients
        self.ids = np.array([patient.id for patient in patients])

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        return self.patients[index]
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.patients):
            raise StopIteration
        else:
            patient = self.patients[self.current_index]
            self.current_index += 1
            return patient
    def travel_time(self, p):

        if len(self) == 0 : return 0

        d = p.d
        time = d[0][self[0].id]
        for i in range(len(self)-1):
            time += get_distance(d, self[i], self[i+1])
        time += d[self[-1].id][0]
        return time

class Solution:
    def __init__(self, p, route_indexes:np.ndarray = None) -> None:
        """ 
        Parameters
        ----------
        Route_indexes : np.ndarray(dtype=int)
            a list of id for the routes
        """
        # If roue_indexes is not provided, build according to problem parameters. Else, use route_indexes
        route_indexes = generate_sub_arrays(p.nbr_patients, p.nbr_nurses) if route_indexes == None else route_indexes
        routes = []
        for route in route_indexes:
            route = Route([p.patients[patient_id] for patient_id in route])
            routes.append(route)
        self.routes = np.array(routes, dtype=object)
        self.matrix = route_indexes

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
    def fitness(self, p):
        return self.travel_time(p)


class Population:
    def __init__(self, pop_size, p, init="random", solutions=None):
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
        return random.sample(self.solutions, k)
    
# Parent selection
def get_parents(population, p):
    pop_size = len(population)
    parents = []
    for i in range(pop_size):
        parent = tournament_selection(population, p, k=3)
        parents.append(parent)
    return Population(pop_size, p, init="custom", solutions=parents)

# Offspring creation

def mutate_population(population:Population,problem:Problem):
    """Takes as input a population, selects parents, applies crossver between different solutions, and then mutates each solution.
    """
    new_pop = get_parents(population, problem)
    solutions = []
    for solution in new_pop :
        new_solution = Solution(problem, mutate_solution(solution))
        solutions.append(new_solution)
    return Population(population.pop_size, population.problem, init='custom', solutions = solutions)

def mutate_solution(s:Solution, n_mutations:int =3):
    """Does a crossover over a random selection of 2 routes from the solution,
    and does `n_mutations` within random roads.
        
    Returns
    -------
        new_route_indexes : list(list(ids))
            A new solution's route_indexes with effectuated random_crossover.
    """
    new_route_indexes = np.copy(s.matrix)

    i,j = np.random.choice(len(new_route_indexes), 2)
    new_route_indexes[i], new_route_indexes[j] = intra_cross_over(new_route_indexes[i], new_route_indexes[j])

    mut_idx = np.random.choice(len(new_route_indexes), n_mutations)
    for idx in mut_idx :
        new_route_indexes[idx] = inverse_mutation(new_route_indexes[idx])

    return new_route_indexes

def inverse_mutation(parent):
    """ Takes as input a route p1 and performs in-place inverse mutation
    """
    L = len(parent)
    # If parent is too short, then don't inverse_mutate
    if L <= 1: 
        return parent
    
    # If length is sufficient, do a mutation
    else :
        i,j = np.random.choice(L, 2, replace=False)
        if i > j:
            i, j = j, i

        # Invert the segment between i and j
        parent[i:j+1] = parent[i:j+1][::-1]

        return parent


def intra_cross_over(p1,p2):
    """Does a crossover, within the routes of a same solution. Is called by intra_cross_over()

    Parameters
    ----------
        p1 : list(int)
            The list of ids of parent 1
        p2 : list(int)
            The list of ids of parent 2
    Returns
    -------
    list(ids),lsit(ids)
        The ids corresponding to the routes of child 1 and child 2
        
    """

    l1, l2 = len(p1), len(p2)

    if l1 == 0 or l2 == 0 : return p1,p2 # If one route is empty, return the parents (crossover impossible)

    c1 = np.empty(l1, dtype=int)
    c2 = np.empty(l2, dtype=int)
    if not l1 > l2 : 
        p1,p2 = p2,p1 # Making sure p1 is the longer route
        c1,c2 = c2, c1
    min_length = min(l1, l2)
    x_over_point = np.random.randint(0,min_length)
    c1[:x_over_point] = p2[:x_over_point]
    c1[x_over_point:] = p1[x_over_point:]
    c2[:x_over_point] = p1[:x_over_point]
    c2[x_over_point:] = p2[x_over_point:]

    return c1, c2

    # # Reconstruct patients
    # c1_patients = [all_patients[np.where(all_ids == id)[0][0]] for id in c1]
    # c2_patients = [all_patients[np.where(all_ids == id)[0][0]] for id in c2]
    
    # # c1_patients = 
    # return Route(c1_patients), Route(c2_patients)

# Offspring surviving