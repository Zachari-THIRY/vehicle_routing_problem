import numpy as np

def get_distance(d, p1, p2):
    """
    Parameters
    ----------
    p1: Patient
        The first patient
    p2: Patient
        The Second patient
    """
    return d[p1.id][p2.id]