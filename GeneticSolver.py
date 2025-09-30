from enum import Enum
import random
from MutantMaze import MutantMaze
import copy

class Gen(Enum):
    """
    Representa los genes posibles (movimientos) que puede realizar el agente.
    Cada gen corresponde a un movimiento en el laberinto:
    - UP: Arriba
    - DOWN: Abajo
    - LEFT: Izquierda
    - RIGHT: Derecha
    """
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Cromosoma:
    """
    Clase que representa un cromosoma (una solución candidata).
    Cada cromosoma está compuesto por una secuencia de genes (movimientos).
    """

    def __init__(self, num_genes):
        """
        Inicializa un cromosoma con una secuencia aleatoria de genes.

        :param num_genes: número total de genes (movimientos) que tendrá el cromosoma.
        """
        self.__genes = [random.choice(list(Gen)) for _ in range(num_genes)]
        self.__fitness = None
        self.__steps_used = 0

    def mutate(self, prob=0.05):
        """
        Aplica mutación a los genes con cierta probabilidad.
        Reemplaza un gen por otro distinto.

        :param prob: probabilidad de mutación por gen.
        """
        for i in range(len(self.__genes)):
            if random.random() < prob:
                opciones = [g for g in Gen if g != self.__genes[i]]
                self.__genes[i] = random.choice(opciones)

    def set_fitness(self, fitness):
        """Asigna un valor de fitness al cromosoma."""
        self.__fitness = fitness

    def get_fitness(self):
        """Devuelve el fitness del cromosoma."""
        return self.__fitness
    
    def get_genes(self):
        """Devuelve la lista de genes (movimientos)."""
        return self.__genes

    def set_genes(self, genes):
        """Asigna una nueva secuencia de genes al cromosoma."""
        self.__genes = genes

    def add_steps(self):
        """Incrementa el contador de pasos usados por el cromosoma."""
        self.__steps_used += 1

    def get_steps_used(self):
        """Devuelve la cantidad de pasos usados."""
        return self.__steps_used


class GeneticSolver:
    """
    Implementación de un Algoritmo Genético para resolver el MutantMaze.
    Genera una población de cromosomas y los evoluciona hasta encontrar una solución.
    """

    def __init__(self, maze: MutantMaze, num_steps=None, poblation=None):
        """
        Inicializa el solver genético.

        :param maze: instancia del laberinto MutantMaze.
        :param num_steps: número máximo de pasos por cromosoma.
        :param poblation: tamaño de la población inicial.
        """
        self.maze = maze
        self.num_steps = num_steps if num_steps is not None else self.calculate_max_steps()
        self.poblation = poblation if poblation is not None else self.estimate_population()
        self.cromosomas = [Cromosoma(self.num_steps) for _ in range(self.poblation)]

    def calculate_max_steps(self):
        """
        Calcula el número máximo de pasos permitidos para un cromosoma,
        dependiendo del tamaño del laberinto, cantidad de muros, mutaciones y salidas.

        :return: número máximo de pasos.
        """
        if self.maze.size <= 20:
            base_steps = self.maze.size * 4
        elif self.maze.size <= 40:
            base_steps = self.maze.size * 6
        else:
            base_steps = self.maze.size * 8 
        
        wall_factor = 1 + (self.maze.wall_prob * 0.8)
        mutation_factor = 1 + (self.maze.mutation_prob * 0.6)
        complexity_factor = 1 + (self.maze.num_exits * 0.1)
        
        steps_adjusted = int(base_steps * wall_factor * mutation_factor * complexity_factor)
        
        min_steps = self.maze.size * 3
        max_steps = self.maze.size * 15
        
        return max(min_steps, min(steps_adjusted, max_steps))
    
    def estimate_population(self, base=20):
        """
        Estima el tamaño de la población inicial en base al tamaño del laberinto,
        cantidad de pasos y densidad de muros.

        :param base: tamaño base de población.
        :return: tamaño estimado de población.
        """
        factor_size = 2
        factor_steps = -0.5
        factor_walls = 50 * self.maze.wall_prob

        estimated = base + int(self.maze.size * factor_size) + int(self.num_steps * factor_steps) + int(factor_walls)
        estimated = max(10, estimated)
        estimated = min(200, estimated)
        return estimated

    def estimate_max_generations(self):
        """
        Estima el número máximo de generaciones a ejecutar.

        :return: número de generaciones.
        """
        k = 3
        generations = int(k * (self.num_steps * self.maze.size) / self.poblation)
        return max(generations, 20)

    def fitness(self, colissions, final_distance, find_false_exits, find_exit, steps, initial_distance):
        """
        Calcula la función de fitness de un cromosoma.

        :param colissions: número de colisiones contra muros.
        :param final_distance: distancia final al objetivo.
        :param find_false_exits: número de salidas falsas encontradas.
        :param find_exit: True si se encontró la salida válida.
        :param steps: cantidad de pasos usados.
        :param initial_distance: distancia inicial a la salida.
        :return: valor de fitness.
        """
        # Caso exitoso: encontró la salida
        if find_exit:
            base_success = 50000
            efficiency_bonus = max(0, (self.num_steps - steps) / self.num_steps) * 20000
            return int(base_success + efficiency_bonus)
        
        # Caso sin éxito: no encontró salida
        MAX_NO_EXIT = 8000
        score = 2000
        
        # 1. Progreso en distancia
        if final_distance < initial_distance:
            improvement = (initial_distance - final_distance) / initial_distance
            score += improvement * 3000
        elif final_distance > initial_distance:
            worsening = (final_distance - initial_distance) / initial_distance
            score -= min(worsening * 1500, 1000)
        
        # 2. Eficiencia de movimiento
        if steps > 0:
            efficiency = (steps - colissions) / steps
            score += efficiency * 2000
        
        # 3. Penalizaciones
        penalties = 0
        penalties += find_false_exits * 200
        penalties += min(colissions * 30, 800)
        penalties += min(final_distance * 8, 1000)
        
        # 4. Bonus por usar pocos pasos
        if final_distance < initial_distance and steps < self.num_steps * 0.5:
            steps_bonus = (1 - steps / self.num_steps) * 1000
            score += steps_bonus
        
        fitness_value = score - penalties
        return max(100, min(int(fitness_value), MAX_NO_EXIT))
    
    def calculate_target_fitness(self):
        """
        Define el valor de fitness objetivo que indica haber encontrado una solución.
        """
        return 30000

    def simulation_maze(self, cromosoma, maze):
        """
        Simula el recorrido de un cromosoma en una copia del laberinto.

        :param cromosoma: cromosoma a evaluar.
        :param maze: copia del laberinto.
        """
        genes = cromosoma.get_genes()
        num_colissions = 0
        find_false_exits = 0
        find_exit = False
        valid_exit_pos = maze.get_valid_exit()
        initial_distance = maze.manhattan_distance(maze.get_agent_position(), valid_exit_pos)

        for gen in genes:
            cromosoma.add_steps()
            ax, ay = maze.get_agent_position()
            mx, my = gen.value
            move = (ax + mx, ay + my)

            if maze.is_valid_position(move):
                maze.move_agent(move)
                current_pos = maze.get_agent_position()
                
                if maze.is_valid_exit(current_pos):
                    find_exit = True
                    break
                elif maze.is_exit(current_pos):
                    find_false_exits += 1
            else:
                num_colissions += 1

            maze.mutate_walls()

        agent_pos = maze.get_agent_position()
        distance_to_exit = maze.manhattan_distance(agent_pos, valid_exit_pos)

        cromosoma.set_fitness(self.fitness(num_colissions, distance_to_exit,
                                           find_false_exits, find_exit, cromosoma.get_steps_used(), initial_distance))

    def crossing_chromosomes(self):
        """
        Realiza la selección y cruce de cromosomas para generar una nueva población.
        Incluye elitismo (se conserva el mejor individuo).
        """
        sorted_cromosomas = sorted(self.cromosomas, key=lambda c: c.get_fitness(), reverse=True)
        offspring = []
        
        if sorted_cromosomas:
            offspring.append(sorted_cromosomas[0])  # elitismo
        
        for i in range(1, len(sorted_cromosomas) - 1, 2):
            padre1 = sorted_cromosomas[i]
            padre2 = sorted_cromosomas[i + 1]
            
            punto_corte = random.randint(1, len(padre1.get_genes()) - 1)
            
            hijo1_genes = padre1.get_genes()[:punto_corte] + padre2.get_genes()[punto_corte:]
            hijo2_genes = padre2.get_genes()[:punto_corte] + padre1.get_genes()[punto_corte:]
            
            hijo1 = Cromosoma(len(hijo1_genes))
            hijo1.set_genes(hijo1_genes)
            hijo1.mutate(prob=0.08)
            
            hijo2 = Cromosoma(len(hijo2_genes))
            hijo2.set_genes(hijo2_genes)
            hijo2.mutate(prob=0.08)
            
            offspring.extend([hijo1, hijo2])
        
        self.cromosomas = offspring[:self.poblation]

    def solveMaze(self, max_generations=None):
        """
        Ejecuta el algoritmo genético hasta encontrar una solución o llegar al límite de generaciones.

        :param max_generations: número máximo de generaciones (opcional).
        :return: mejor cromosoma encontrado y el estado del laberinto que recorrio.
        """
        self.max_generations = max_generations if max_generations is not None else self.estimate_max_generations()
        target_fitness = self.calculate_target_fitness()

        best_overall = None
        best_maze_state = None
        best_fitness = -1

        for generation in range(self.max_generations):
            chromosome_maze_pairs = []
            
            for cromosoma in self.cromosomas:
                maze_copy = copy.deepcopy(self.maze)
                self.simulation_maze(cromosoma, maze_copy)
                chromosome_maze_pairs.append((cromosoma, maze_copy))

            chromosome_maze_pairs.sort(key=lambda pair: pair[0].get_fitness(), reverse=True)
            self.cromosomas = [pair[0] for pair in chromosome_maze_pairs]

            best_current, current_maze_state = chromosome_maze_pairs[0]
            current_fitness = best_current.get_fitness()
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_overall = best_current
                best_maze_state = copy.deepcopy(current_maze_state)

            if current_fitness >= target_fitness:
                actual_pos = current_maze_state.get_agent_position()
                if current_maze_state.is_valid_exit(actual_pos):
                    return best_overall, current_maze_state

            if generation < self.max_generations - 1:
                self.crossing_chromosomes()

        return best_overall, best_maze_state
