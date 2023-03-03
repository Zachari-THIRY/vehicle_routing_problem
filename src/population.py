import numpy as np

class Route:
    def __init__(self, patients) -> None:
        self.patients = patients
        self.ids = [patient.id for patient in patients]

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

class Population:
    def __init__(self, pop_size, p):
        self.solutions = [Solution(p) for _ in range(pop_size)]
    def __getitem__(self, index):
        return self.solutions[index]

def generate_sub_arrays(N,n):
    arr = np.arange(1, N+1)
    permuted_arr = np.random.permutation(arr)

    indices = np.random.choice(np.arange(2, N+1), size=n, replace=False)
    indices.sort()

    sub_arrays = np.split(permuted_arr, indices)
    return np.array(sub_arrays, dtype=object)