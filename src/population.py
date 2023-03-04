import numpy as np
from utils import get_distance, generate_sub_arrays, tournament_selection
import random

class Route:
    def __init__(self, patients) -> None:
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
    def __init__(self, p) -> None:
        route_indexes = generate_sub_arrays(p.nbr_patients, p.nbr_nurses)
        routes = []
        for route in route_indexes:
            route = Route([p.patients[patient_id] for patient_id in route])
            routes.append(route)
        self.routes = routes
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
    

def get_parents(population, p):
    pop_size = len(population)
    parents = []
    for i in range(pop_size):
        parent = tournament_selection(population, p, k=3)
        parents.append(parent)
    return Population(pop_size, p, init="custom", solutions=parents)