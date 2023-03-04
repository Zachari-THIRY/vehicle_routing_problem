import numpy as np
import random

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

def tournament_selection(population,p, k=3):
    tournament = population.sample(k)
    winner = min(tournament, key=lambda x: x.fitness(p)) # Calculate fitness using fitness_func
    return winner