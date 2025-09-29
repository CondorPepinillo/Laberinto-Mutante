import MutantMaze
from AStarSolver import AStarSolver
from GeneticSolver import GeneticSolver
import time

# Ejemplo de uso
size = 10
mutation_prob=0.1
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 10
mutation_prob=0.9
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 25
mutation_prob=0.1
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 25
mutation_prob=0.9
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 50
mutation_prob=0.1
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 50
mutation_prob=0.9
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 100
mutation_prob=0.1
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")

# Ejemplo de uso
size = 100
mutation_prob=0.9
wall_prob=0.3
num_exits=3

maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)

#Solucion Algoritmos Genetico
solver = GeneticSolver(maze)

inicio = time.perf_counter()
solver.solveMaze()
fin = time.perf_counter()

time_genetic = fin - inicio 

print(f"Genetic;{size};{mutation_prob};{time_genetic:.4f}")

#Solucion Algotimos A*
solver = AStarSolver(maze)

inicio = time.perf_counter()
solver.solve()
fin = time.perf_counter()

time_AStar = fin - inicio

print(f"A*;{size};{mutation_prob};{time_AStar:.4f}")