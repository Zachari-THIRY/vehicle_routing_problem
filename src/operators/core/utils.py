import numpy as np
import matplotlib.pyplot as plt
def get_distance(d, p1, p2):
    """
    Returns the travel_time between p1 and p2. If p1 or p2 aren't specified, then the depot is considered.

    Parameters
    ----------
    p1: Patient
        The first patient
    p2: Patient
        The Second patient

    Returns
    -------
    t : int
        The travel time between p1 and p2
    """

    pid1 = 0 if p1 == None else p1.id
    pid2 = 0 if p2 == None else p2.id
    return d[pid1][pid2]

def round_robin_init(p):
    """
    From a problem p, generates a set of patient routes order based on their time windows.
    
    Parameters
    ----------
    p : Problem
        The problem at hand
    Returns
    -------
    route_indexes : np.ndarray
        A set of route_indexes with smart initialisation.
    """
    pids = p.patients.keys()

    sorted_pids = sorted(pids, key=lambda x : p.patients[x].start_time)
    arr = np.empty(p.nbr_nurses, dtype=np.ndarray)
    for i in range(len(arr)):
        sub_arr = np.array([sorted_pids[j] for j in range(i, len(sorted_pids), len(arr))])
        arr[i] = sub_arr
    return np.array(arr)

def display_solution(solution, problem, filename="fig.ping"):
    
    # Scattering the patients in background : 
    x_depot,y_depot = problem.depot.x, problem.depot.y
    patients_coordinates = [(problem.patients[pid].x, problem.patients[pid].y) for pid in problem.patients]
    X = [patient_coordinate[0] for patient_coordinate in patients_coordinates]
    Y = [patient_coordinate[1] for patient_coordinate in patients_coordinates]
    plt.scatter(X,Y)
    plt.scatter(x_depot,y_depot, color='black', marker='o')

    for row in solution.matrix:
        if len(row) == 0 : continue
        # Adding depot
        row_points = [(x_depot, y_depot)]
        # Adding patients
        for pid in row:
            row_points.append((problem.patients[pid].x, problem.patients[pid].y))
        # Adding depot
        row_points.append((x_depot, y_depot))
        row_points = np.array(row_points)
        X,Y = row_points[:,0], row_points[:,1]
        plt.plot(X,Y)

    plt.savefig(filename)

def display_round_info(population, problem, epoch:int):
    """
    Displays information about the current epoch.
    Parameters
    ----------
    population : Population
    problem : Problem
    epoch : int or str
    """
    assert type(epoch) == int or epoch == "end", "Parameter `epoch` must be either an int or `end`"
    if epoch == "end":
        print("##############__End__################")
    else : 
        print("#####################################")
        print(f"""Current status: epoch {epoch:03d}""")
    print("late_time:", np.mean([sol.late_time(problem) for sol in population]))
    print("travel_time:", np.mean([sol.travel_time(problem) for sol in population]))
    print("min_late_time:", np.min([sol.late_time(problem) for sol in population]))
    print("min_travel_time:", np.min([sol.travel_time(problem) for sol in population]))
