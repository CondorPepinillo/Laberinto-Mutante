import heapq
from typing import List, Tuple, Dict, Optional
from MutantMaze import MutantMaze

class AStarSolver:
    def __init__(self, maze: MutantMaze):
        self.maze = maze
        self.path = []
        self.steps = 0
        self.blacklisted_exits = set()  # Conjunto para guardar salidas inválidas ya probadas, solo una salida es correcta 'E'
        
    def solve(self) -> bool:
        """Resuelve el laberinto mutante usando A*, probando diferentes salidas."""
        while True:
            # Recalculamos el camino óptimo desde la posición actual
            current_path = self._a_star_search()
            
            if not current_path:
                print("No se encontró camino válido. Esperando un turno...")
                # Las paredes mutan aunque no nos movamos
                self.maze.mutate_walls()
                self.steps += 1
                
                # Mostramos información
                print(f"Paso {self.steps}: Esperando (sin movimiento)")
                self.maze.print_maze()
                continue  # Volver a intentar en el siguiente turno
                
            # Movemos al agente al siguiente paso en el camino
            next_pos = current_path[1]  # [0] es la posición actual
            
            # Antes de movernos, las paredes pueden mutar
            self.maze.mutate_walls()
            
            # Verificamos que el movimiento sigue siendo válido (por si mutó)
            if not self.maze.is_valid_position(next_pos):
                print("¡La mutación bloqueó nuestro camino! Recalculando...")
                continue
                
            # Movemos al agente
            self.maze.move_agent(next_pos)
            self.steps += 1
            
            # Mostramos información
            print(f"Paso {self.steps}: Moviendo a {next_pos}")
            self.maze.print_maze()
            
            # Si llegamos a una salida
            if self.maze.is_exit(self.maze.get_agent_position()):
                if self.maze.is_valid_exit(self.maze.get_agent_position()):
                    print(f"¡Salida válida encontrada en {self.steps} pasos!")
                    return True
                else:
                    # Añadir esta salida a la blacklist
                    exit_pos = self.maze.get_agent_position()
                    self.blacklisted_exits.add(exit_pos)
                    print(f"¡Salida inválida encontrada en {exit_pos}! Añadiendo a blacklist.")
                    # No salimos, continuamos buscando
            
    def _a_star_search(self) -> List[Tuple[int, int]]:
        """Implementación del algoritmo A* que ignora salidas en la blacklist."""
        start = self.maze.get_agent_position()
        
        # Buscamos cualquier salida que no esté en la blacklist
        possible_exits = []
        for exit_pos in self.maze.exits:
            if exit_pos not in self.blacklisted_exits:
                possible_exits.append(exit_pos)
                
        # Usamos A* para cada salida posible y escogemos el camino más corto
        best_path = None # Para inicializar
        best_length = float('inf') # Para inicializar, asi cualquier camino es mejor que este en un inicio
        
        for goal in possible_exits:
            # Estructuras para el algoritmo
            open_set = []
            heapq.heappush(open_set, (0, start))
            
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
            g_score: Dict[Tuple[int, int], float] = {start: 0}
            f_score: Dict[Tuple[int, int], float] = {start: self.maze.manhattan_distance(start, goal)}
            
            open_set_hash = {start}
            
            while open_set:
                current = heapq.heappop(open_set)[1]
                open_set_hash.remove(current)
                
                if current == goal:
                    # Reconstruir el camino
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()
                    
                    # Verificar si es el mejor camino hasta ahora
                    if len(path) < best_length:
                        best_path = path
                        best_length = len(path)
                    break
                    
                for neighbor in self.maze.get_neighbors(current):
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.maze.manhattan_distance(neighbor, goal)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
            return best_path

# Ejemplo de uso
if __name__ == "__main__":
    # Crear laberinto
    maze = MutantMaze(size=20, wall_prob=0.3, mutation_prob=0.1, num_exits=3)
    
    print("Laberinto inicial:")
    print(f"Salida válida: {maze.valid_exit}")
    maze.print_maze()
    
    # Resolver
    solver = AStarSolver(maze)
    solver.solve()
