import MutantMaze
from AStarSolver import AStarSolver
from GeneticSolver import GeneticSolver
import time

# Ejemplo de uso
size =20
mutation_prob=0.1
wall_prob=0.3
num_exits=3

print("Laberinto:")
maze = MutantMaze.MutantMaze(size, wall_prob, mutation_prob, num_exits)
maze.print_maze()

#Solucion Algoritmos Genetico
print("Algoritmo genetico")
maze.print_maze()
solver = GeneticSolver(maze)
x, y = solver.solveMaze()
print("Resultado:")
y.print_maze()

#Solucion Algotimos A*
print("Algoritmo A*")
maze.print_maze()
solver = AStarSolver(maze)
solver.solve()
print("Resultado:")
maze.print_maze()
