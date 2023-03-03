import json

class Problem:
    def __init__(self, filePath: str):
        # Load data from JSON file
        with open(filePath, 'r') as f:
            data = json.load(f)
        
        # Assign attributes
        self.instance_name = data["instance_name"]
        self.nbr_nurses = data["nbr_nurses"]
        self.Capacity_nurse = data["capacity_nurse"]
        self.nbr_patients = len(data["patients"])
        self.patients = {Patient(id, data["patients"][id]) for id in data["patients"]}
        self.depot = Depot(data["depot"])


class Depot:
    def __init__(self, info) -> None:
        self.return_time = info["return_time"]
        self.x = info["x_coord"]
        self.y = info["y_coord"]
class Patient:
    def __init__(self, id, info) -> None:
        self.id = id
        self.x = info["x_coord"]
        self.y = info["y_coord"]
        self.demand = info["demand"]
        self.start_time = info["start_time"]
        self.end_time = info["end_time"]
        self.care_time = info["care_time"]