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

    def get_route_from_ids(ids, p):
        patients = [p.patients[id] for id in ids]
        return Route(patients)

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
    """
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
    assert len(population) %2 == 0, "Population length must be even"

    new_pop = get_parents(population, problem)
    old_solutions = population.solutions
    xov_solutions = [] # list of route_indexes corresponding to solutions. Has to be converted to type Solution later.

    # Crossover between parents
    for i in range(len(new_pop)//2):
        children = extra_cross_over(new_pop[i].matrix ,new_pop[i+1].matrix)
        for child in children : 
            xov_solutions.append(child)

    # Mutate within solutions
    mutated_solutions_idx = []
    for new_solution in new_pop :
        new_solution_idx = mutate_solution(new_solution)
        mutated_solutions_idx.append(new_solution_idx)

    # Transform list of route_indexes in list of solutions
    new_solutions = [Solution(problem, route_indexes) for route_indexes in mutated_solutions_idx]
    
    # Pure elitism : 
    sorted_solutions = sorted(new_solutions + old_solutions, key=lambda x: x.fitness(problem), reverse=False)[0:16]
    return Population(population.pop_size, population.problem, init='custom', solutions = sorted_solutions)

# TODO : Check that it actualy performs n_mutations
def mutate_solution(s:Solution, n_mutations:int = 3):
    """Does a crossover over a random selection of 2 routes from the solution,
    and does `n_mutations` within random roads.

    Parameters
    ----------
    s : Solution
        The solution to be mutated
    n_mutations : int
        The number of mutations to apply
        
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

def extra_cross_over(p1, p2):
    """
    Selects a random splitting point (same for each parent), keeps the first part, 
    and then completes if possible with elements from the other parent.
    Parameters
    ----------
    p1 : np.ndarray
        The matrix of parent solution 1
    p2 : np.ndarray
        The patrix of parent solution 2
    """
  

    # Gathering the ids in order and lengths

    p1_ids = np.concatenate([np.fromiter(subarr, dtype=int) for subarr in p1]).tolist()
    p2_ids = np.concatenate([np.fromiter(subarr, dtype=int) for subarr in p2]).tolist()

    p1_lengths = [len(row) for row in p1]
    p2_lengths = [len(row) for row in p2]

    # Finding crossover point
    x_over_point = np.random.randint(1, len(p1_ids))

    # Splitting
    p1_head = p1_ids[:x_over_point]
    p1_tail = p1_ids[x_over_point:]
    p2_head = p2_ids[:x_over_point]
    p2_tail = p2_ids[x_over_point:]

    c1_ids = np.concatenate((p1_head, p2_tail), axis=0)
    c2_ids = np.concatenate((p2_head, p1_tail), axis=0)

    c1 = np.empty(len(p1_lengths), dtype=object)
    c2 = np.empty(len(p2_lengths), dtype=object)

    i1, i2 = 0,0
    for i,l in enumerate(p1_lengths):
        new_row =  c1_ids[i1:i1+l]
        c1[i] = new_row
        i1 += l
    for i,l in enumerate(p2_lengths):
        new_row =  c2_ids[i1:i1+l]
        c2[i] = new_row
        i2 += l    
    
    return c1, c2

def appendix_cross_over(p1,p2, problem):
    """
    Parameters
    ----------
    p1 : np.ndarray
        The matrix of parent 1
    p2 : np.ndarray
        The matrix of parent 2

    Returns
    -------

    c1, c2 : (np.ndarray, np.ndarray)
        The offspring crossovers
    """
    assert len(p1) == len(p2), "Matrixes `p1` and `p2` should have the same number of rows"
    L = len(p1)
    pop = np.random.randint(1, L, 2)
    
    to_pop_pids = p1[pop[0]], p2[pop[1]] # Contains the routes to be popped

    # Popping the routes in the other solution:
    for index in range(L) :
        p1[index] = np.setdiff1d(p1[index], to_pop_pids[1])
        p2[index] = np.setdiff1d(p2[index], to_pop_pids[0])

    # Re-inserting patients into the routes : 
    np.random.shuffle(to_pop_pids[0])
    np.random.shuffle(to_pop_pids[1])

    for pid in to_pop_pids[1]:
        p1 = smart_insert(p1, pid, problem)
    for pid in to_pop_pids[0]:
        p2 = smart_insert(p2, pid, problem)

    return p1, p2
                

def smart_insert(parent, pid, p:Problem) -> np.ndarray:
    """
    Parameters
    ----------
    parent : np.ndarray
        Matrix representation of parent
    """
    lengths = np.array([len(row) for row in parent])
    lengths += 1

    # Build insertion penalties matrix

    insert_penalties = []
    for row in parent:
        row_penalties = []
        baseline = Route.get_route_from_ids(row, p).travel_time(p)
        for i in range(len(row) + 1):
            new_row = dumb_insert(row, pid, index = i)
            new_route = Route.get_route_from_ids(new_row, p)
            row_penalties.append(new_route.travel_time(p)-baseline)
        insert_penalties.append(row_penalties)

    # Retrieve best position
    min_value, min_index = min(
    (val, (i, j))
    for i, row in enumerate(insert_penalties)
    for j, val in enumerate(row)
    )
    if min_value <= 0 : 
        print("Minimum value:", min_value)
        print("Index of minimum value:", min_index)
        print("Route:", parent[min_index[0]])
        print("Insert pid:", pid)

    parent[min_index[0]] = dumb_insert(row = parent[min_index[0]], pid = pid, index = min_index[1])
    return parent

def dumb_insert(row, pid:int, index:int) -> np.ndarray:
    """Inserts a pid in the specified index"""
    new_row = np.insert(row, index, pid, axis=0)
    return new_row

def fitness_from_matrix(solution_matrix, problem):
    solution = Solution(p = problem, route_indexes=solution_matrix)
    return solution.fitness(problem)

# Offspring surviving