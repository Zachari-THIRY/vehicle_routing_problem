import numpy as np

from utils import get_distance

class Route:
    """
    self.patients : python list of dtype Patient
    self.ids : np.array of ids (used for computation, not representation)
    """
    def __init__(self, patients) -> None:
        """
        Takes as input : patients, the array of dtype Patient
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
    
    def late_time(self, p) -> int:
        if len(self) == 0 : return 0
        d = p.d

        t = 0
        late_time = 0
        prev_patient = None
        for patient in self:
            # Travelling there
            t += get_distance(d, prev_patient, patient)
            # Waiting before care if needed
            if t < patient.start_time :
                t  = patient.start_time
            # Caring for the patient
            t += patient.care_time
            # Checking if care is finished in time
            if t > patient.end_time:
                late_time += t - patient.end_time
            prev_patient = patient
        # Adding time to come back to the depot
        t += get_distance(d, prev_patient, None)
        if t > p.depot.return_time:
            late_time += t-p.depot.return_time
        return late_time

def get_route_from_ids(ids, p):
        patients = [p.patients[id] for id in ids]
        return Route(patients)