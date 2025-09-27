from enum import Enum
import random
from MutantMaze import MutantMaze
import copy

class Gen(Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Cromosoma:
    def __init__(self, num_genes):
        self.__genes = [random.choice(list(Gen)) for _ in range(num_genes)]
        self.__fitness = None
        self.__steps_used = 0

    def mutate(self, prob=0.05):
        for i in range(len(self.__genes)):
            if random.random() < prob:
                opciones = [g for g in Gen if g != self.__genes[i]]
                self.__genes[i] = random.choice(opciones)

    def set_fitness(self, fitness):
        self.__fitness = fitness

    def get_fitness(self):
        return self.__fitness
    
    def get_genes(self):
        return self.__genes

    def set_genes(self, genes):
        self.__genes = genes

    def add_steps(self):
        self.__steps_used +=1

    def get_steps_used(self):
        return self.__steps_used

class GeneticSolver:
    def __init__(self, maze: MutantMaze, num_steps=None, poblation=None):
        self.maze = maze
        self.num_steps = num_steps if num_steps is not None else self.calculate_max_steps()
        self.poblation = poblation if poblation is not None else self.estimate_population()
        self.cromosomas = [Cromosoma(self.num_steps) for _ in range(self.poblation)]

    def calculate_max_steps(self):
        base_steps = self.maze.size * 3
        steps_adjusted = int(base_steps * (1 + self.maze.wall_prob) * (1 + self.maze.mutation_prob))
        return max(steps_adjusted, self.maze.size * 2)
    
    def estimate_population(self, base=20):
        factor_size = 2
        factor_steps = -0.5
        factor_walls = 50 * self.maze.wall_prob

        estimated = base + int(self.maze.size * factor_size) + int(self.num_steps * factor_steps) + int(factor_walls)
        estimated = max(10, estimated)
        estimated = min(200, estimated)
        return estimated

    def estimate_max_generations(self):
        k = 3
        generations = int(k * (self.num_steps * self.maze.size) / self.poblation)
        return max(generations, 20)

    def fitness(self, colissions, distance, find_false_exits, find_exit, steps):
        max_score = self.maze.size * 100
        bonus_exit = max_score * (1 + self.maze.mutation_prob) if find_exit else 0

        penalty_colissions = colissions * (2 * self.maze.size) * (1 - self.maze.wall_prob)
        penalty_false_exits = find_false_exits * (0.5 * self.maze.size)
        penalty_distance = distance * 5
        penalty_steps = int((steps / self.num_steps) * (max_score * 0.3))

        fitness = bonus_exit + max_score - (
            penalty_colissions + penalty_false_exits + penalty_distance + penalty_steps
        )
        return max(1, int(fitness))

    def simulation_maze(self, cromosoma, maze):
        genes = cromosoma.get_genes()
        num_colissions = 0
        find_false_exits = 0
        find_exit = False

        for gen in genes:
            cromosoma.add_steps()
            ax, ay = maze.get_agent_position()
            mx, my = gen.value
            move = (ax + mx, ay + my)

            if maze.is_valid_position(move):
                maze.move_agent(move)
                if not maze.is_valid_exit(maze.get_agent_position()):
                    find_false_exits += 1
                else:
                    print("Salida encontrada")
                    find_exit = True
                    break
            else:
                num_colissions += 1

            maze.mutate_walls()

        agent_pos = maze.get_agent_position()
        valid_exit_pos = maze.get_valid_exit()
        distance_to_exit = maze.manhattan_distance(agent_pos, valid_exit_pos)

        cromosoma.set_fitness(self.fitness(num_colissions, distance_to_exit,
                                           find_false_exits, find_exit, cromosoma.get_steps_used()))

    def crossing_chromosomes(self):
        sorted_cromosomas = sorted(self.cromosomas, key=lambda c: c.get_fitness(), reverse=True)
        offspring = []

        start_idx = 0
        if len(sorted_cromosomas) % 2 != 0:
            offspring.append(sorted_cromosomas[0])
            start_idx = 1

        for i in range(start_idx, len(sorted_cromosomas) - 1, 2):
            padre1 = sorted_cromosomas[i]
            padre2 = sorted_cromosomas[i + 1]

            punto_corte = random.randint(1, len(padre1.get_genes()) - 1)

            hijo1_genes = padre1.get_genes()[:punto_corte] + padre2.get_genes()[punto_corte:]
            hijo2_genes = padre2.get_genes()[:punto_corte] + padre1.get_genes()[punto_corte:]

            hijo1 = Cromosoma(len(hijo1_genes))
            hijo1.set_genes(hijo1_genes)
            hijo1.mutate()

            hijo2 = Cromosoma(len(hijo2_genes))
            hijo2.set_genes(hijo2_genes)
            hijo2.mutate()

            offspring.extend([hijo1, hijo2])

        self.cromosomas = offspring[:self.poblation]

    def solveMaze(self, max_generations=None):
        self.max_generations = max_generations if max_generations is not None else self.estimate_max_generations()

        max_score = self.maze.size * 100
        fitness_max = max_score * (2 + self.maze.mutation_prob)
        target_fitness = int(fitness_max * 0.95)

        best_overall = None
        best_maze_state = None
        best_fitness = -1

        for generation in range(self.max_generations):
            print(f"\n--- Generación {generation+1} ---")

            # Lista para guardar pares (cromosoma, laberinto_simulado)
            chromosome_maze_pairs = []
            
            for cromosoma in self.cromosomas:
                # Crear una copia del laberinto original para cada simulación
                maze_copy = copy.deepcopy(self.maze)
                self.simulation_maze(cromosoma, maze_copy)
                chromosome_maze_pairs.append((cromosoma, maze_copy))

            # Ordenar por fitness (mejor primero)
            chromosome_maze_pairs.sort(key=lambda pair: pair[0].get_fitness(), reverse=True)
            
            # Actualizar lista de cromosomas
            self.cromosomas = [pair[0] for pair in chromosome_maze_pairs]

            # Mejor de esta generación
            best_current, current_maze_state = chromosome_maze_pairs[0]
            current_fitness = best_current.get_fitness()
            print(f"Mejor fitness: {current_fitness}")

            # Actualizar mejor global si es necesario
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_overall = best_current
                best_maze_state = copy.deepcopy(current_maze_state)  # Guardar copia del laberinto
                print("¡Nuevo mejor global encontrado!")

            # Mostrar el laberinto del mejor de esta generación
            print(f"\nLaberinto recorrido por el mejor cromosoma (Generación {generation+1}):")
            current_maze_state.print_maze()
            print(f"Posición final del agente: {current_maze_state.get_agent_position()}")
            print(f"Salida válida: {current_maze_state.get_valid_exit()}")

            # Condición de parada si alcanzó fitness objetivo
            if current_fitness >= target_fitness:
                print("\n✅ Solución encontrada!")
                print(f"Número de generaciones: {generation+1}")
                return best_overall, best_maze_state

            # Generar nueva población
            self.crossing_chromosomes()

        print(f"\n⚠️ Mejor salida encontrada en el número máximo de generaciones ({self.max_generations})")
        if best_maze_state:
            print("\nLaberinto final del mejor cromosoma global:")
            best_maze_state.print_maze()
            print(f"Posición final del agente: {best_maze_state.get_agent_position()}")
            print(f"Salida válida: {best_maze_state.get_valid_exit()}")
            print(f"Fitness alcanzado: {best_fitness}")
        return best_overall, best_maze_state

# Ejemplo de uso
if __name__ == "__main__":
    # Crear laberinto
    maze = MutantMaze(size=25, wall_prob=0.3, mutation_prob=0.1, num_exits=3)
    
    print("Laberinto inicial:")
    print(f"Salida válida: {maze.valid_exit}")
    print(f"Posición inicial del agente: {maze.get_agent_position()}")
    maze.print_maze()
    
    # Resolver
    solver = GeneticSolver(maze)
    best_cromosoma, best_maze = solver.solveMaze() #Intenta buscar la mejor solucion
    print(f"Cantidad de pasos: {best_cromosoma.get_steps_used()}")


    #    def solveMaze(self):
#        while True:
#            for cromosoma in self.cromosomas:
#                maze_copy = copy.deepcopy(self.maze)
#                self.simulation_maze(cromosoma, maze_copy)
#                if cromosoma.get_fitness() > 1500:
#                    maze_copy.print_maze()
#                    return True
#            self.crossing_chromosomes()

#    def fitness(self, colissions, distance, find_false_exits, find_exit, steps):
#        """
#        Calcula el fitness del cromosoma.
#        - find_exit: 1000 si llegó a la salida, 0 si no
#        - colissions: número de choques contra muros
#        - find_false_exits: número de salidas falsas visitadas
#        - distance: distancia Manhattan a la salida al final
#        - steps: cantidad de pasos usados por el cromosoma
#        """
#        # Penalizamos colisiones y falsos exits
#        penalizacion = colissions * 30 + find_false_exits * 10 + steps
#        
#        # Fitness final
#        fitness = find_exit + max(0, 1000 - distance - penalizacion)
#        return fitness