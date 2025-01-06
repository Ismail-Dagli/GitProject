import numpy as np
from itertools import combinations

# Gegebene Prüfmatrix H
H = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1]
])

# Funktion, um den Mindest-Hamming-Abstand basierend auf linear abhängigen Spalten zu berechnen
def min_hamming_distance(H):
    n = H.shape[1]  # Anzahl der Spalten
    for r in range(2, n + 1):  # Anzahl der zu prüfenden Spaltenkombinationen
        for cols in combinations(range(n), r):
            submatrix = H[:, cols]
            if np.linalg.matrix_rank(submatrix) < r:
                return r
    return n

# Mindest-Hamming-Abstand berechnen
d_min = min_hamming_distance(H)
print(f"Der Mindest-Hamming-Abstand ist: {d_min}")
