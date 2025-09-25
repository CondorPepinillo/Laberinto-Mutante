import numpy as np
import random
from typing import List, Tuple


class MutantMaze:
    def __init__(self, size, wall_prob, mutation_prob, num_exits): #example: size=10, wall_prob=0.3, mutation_prob=0.1, num_exits=3
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
        self.agent_pos = (size // 2, size // 2)
        
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

        # Obtener todas las posiciones de paredes y vacías, 1 son paredes, 0 son vacías
        walls = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 1]
        empty = [(i, j) for i in range(self.size) for j in range(self.size) 
                 if self.grid[i, j] == 0 and (i, j) != self.agent_pos and (i, j) not in self.exits]
        
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
    
    #distancia que se recorrería entre dos puntos moviéndose en una cuadrícula, no en línea recta como la distancia euclidiana
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula la distancia Manhattan entre dos posiciones."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def print_maze(self):
        """Imprime el laberinto con opción de mostrar la posición del agente."""
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.agent_pos:
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
    
    def move_agent(self, new_pos: Tuple[int, int]) -> bool:
        """Mueve al agente a una nueva posición si es válida. Retorna True si se mueve a la salida válida."""
        if self.is_valid_position(new_pos):
            self.agent_pos = new_pos
            return self.is_valid_exit(new_pos)
        return False

    def get_agent_position(self) -> Tuple[int, int]:
        """Retorna la posición actual del agente."""
        return self.agent_pos

    #Lo puse pero quiza no se use al ya tener para reconocer vecinos
    def get_item_in_cell(self, pos: Tuple[int, int]) -> int:
        """Retorna el contenido de una celda específica."""
        i, j = pos
        return self.grid[i, j]
    

    #Copie el codigo en mutate_walls para esto, puede que lo cambie en mutate walls para usar esto
    def get_walls(self) -> List[Tuple[int, int]]:
        """Retorna una lista de todas las posiciones de paredes."""
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 1]

if __name__ == "__main__":
    # Ejemplo de uso
    mm = MutantMaze(size=10, wall_prob=0.3, mutation_prob=0.1, num_exits=3)
    mm.print_maze()
    mm.mutate_walls()
    mm.print_maze()
    mm.mutate_walls()
    mm.print_maze()
    mm.mutate_walls()
    mm.print_maze()


    print("Posición del agente:", mm.get_agent_position())
    print("Vecinos válidos del agente:", mm.get_neighbors(mm.get_agent_position()))