import numpy as np
import random
from typing import List, Tuple


class MutantMaze:
    def __init__(self, size, wall_prob, mutation_prob, num_exits):
        """
        Inicializa el laberinto mutante.
        
        :param size: Tamaño del laberinto (N x N)
        :param wall_prob: Probabilidad de que una celda sea pared
        :param mutation_prob: Probabilidad de que una pared se mueva en cada paso
        :param num_exits: Número de salidas (solo una es válida)
        """
        self.size = size
        self.wall_prob = wall_prob
        self.mutation_prob = mutation_prob
        self.num_exits = num_exits
        
        # Inicializar laberinto
        self.grid = np.zeros((size, size), dtype=int)
        self._generate_maze()
        
        # Posición inicial fija (centro del laberinto)
        self.start_pos = (size // 2, size // 2)
        
        # Generar salidas
        self._generate_exits()
        
        # Salida válida (aleatoria entre las generadas)
        self.valid_exit = random.choice(self.exits)
        
    def _generate_maze(self):
        """Genera las paredes iniciales del laberinto."""
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.wall_prob:
                    self.grid[i, j] = 1  # 1 representa pared
    
    def _generate_exits(self):
        """Genera las salidas en los bordes del laberinto."""
        self.exits = []
        edge_positions = []
        
        # Generar todas las posiciones de los bordes
        for i in [0, self.size - 1]:
            for j in range(self.size):
                edge_positions.append((i, j))
        
        for j in [0, self.size - 1]:
            for i in range(1, self.size - 1):
                edge_positions.append((i, j))
                
        # Seleccionar num_exits posiciones aleatorias sin repetición
        for pos in random.sample(edge_positions, min(self.num_exits, len(edge_positions))):
            self.grid[pos] = 2  # 2 representa salida
            self.exits.append(pos)
    
    def mutate_walls(self):
        """Mueve algunas paredes según la probabilidad de mutación."""
        walls = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 1]
        empty = [(i, j) for i in range(self.size) for j in range(self.size) 
                 if self.grid[i, j] == 0 and (i, j) != self.start_pos and (i, j) not in self.exits]
        
        for wall_pos in walls:
            if random.random() < self.mutation_prob and empty:
                # Mover esta pared a una posición vacía
                new_pos = random.choice(empty)
                self.grid[wall_pos] = 0
                self.grid[new_pos] = 1
                empty.remove(new_pos)
                empty.append(wall_pos)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es válida (dentro del laberinto y no es pared)."""
        i, j = pos
        return 0 <= i < self.size and 0 <= j < self.size and self.grid[i, j] != 1
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Obtiene las posiciones vecinas válidas."""
        i, j = pos
        neighbors = []
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (i + di, j + dj)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
                
        return neighbors
    
    def is_exit(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es una salida."""
        return pos in self.exits
    
    def is_valid_exit(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es la salida válida."""
        return pos == self.valid_exit
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula la distancia Manhattan entre dos posiciones."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def print_maze(self, agent_pos: Tuple[int, int] = None):
        """Imprime el laberinto con opción de mostrar la posición del agente."""
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == agent_pos:
                    row.append('A')
                elif (i, j) == self.valid_exit:
                    row.append('E')
                elif (i, j) in self.exits:
                    row.append('e')
                elif self.grid[i, j] == 1:
                    row.append('#')
                else:
                    row.append('.')
            print(' '.join(row))
        print()
        

mm = MutantMaze(size=10, wall_prob=0.3, mutation_prob=0.1, num_exits=3)
mm.print_maze(agent_pos=mm.start_pos)
mm.mutate_walls()
mm.print_maze(agent_pos=mm.start_pos)
mm.mutate_walls()
mm.print_maze(agent_pos=mm.start_pos)
mm.mutate_walls()
mm.print_maze(agent_pos=mm.start_pos)
