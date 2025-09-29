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

    def fitness(self, colissions, final_distance, find_false_exits, find_exit, steps, initial_distance):
        """
        Versión corregida - initial_distance para medir PROGRESO REAL
        """
        BASE_SCORE = self.maze.size * 100
        
        # ✅ BONUS POR ENCONTRAR SALIDA
        if find_exit:
            success_bonus = BASE_SCORE * 3
            efficiency_bonus = (1 - steps / self.num_steps) * BASE_SCORE
            return int(success_bonus + efficiency_bonus)
        
        # 📈 RECOMPENSA POR PROGRESO EN DISTANCIA
        progress_bonus = 0
        
        # ¿Me acerqué a la salida?
        if final_distance < initial_distance:
            distance_saved = initial_distance - final_distance
            progress_ratio = distance_saved / initial_distance  # % de mejora
            progress_bonus = progress_ratio * BASE_SCORE * 2
        
        # 🎯 EFICIENCIA DE MOVIMIENTO
        movement_efficiency = 0
        if steps > 0:
            # Recompensa por menos colisiones
            collision_ratio = (steps - colissions) / steps
            movement_efficiency = collision_ratio * BASE_SCORE * 0.5
        
        # 🚫 PENALIZACIONES
        penalties = 0
        
        # Penalización por salidas falsas (grave error)
        penalties += find_false_exits * 50
        
        # Penalización por distancia residual
        penalties += final_distance * 3
        
        # Penalización por muchas colisiones
        penalties += colissions * 8
        
        # Cálculo final
        fitness_value = BASE_SCORE + progress_bonus + movement_efficiency - penalties
        return max(1, int(fitness_value))
    
    def calculate_target_fitness(self):
        """
        Calcula el objetivo basado en la función fitness ACTUAL
        que premia encontrar la salida con BASE_SCORE * 3 + efficiency_bonus
        """
        BASE_SCORE = self.maze.size * 100
        
        # ✅ Fitness MÍNIMO cuando encuentra salida (peor caso)
        # - Encuentra salida en el ÚLTIMO paso posible
        # - Máximo de colisiones y salidas falsas
        min_success_fitness = BASE_SCORE * 3  # Solo el success_bonus básico
        min_success_fitness = int(min_success_fitness * 0.85)  # 85% del mínimo teórico
        
        # ✅ Fitness ESPERADO cuando encuentra salida (caso promedio)
        # - Encuentra salida en ~50% de los pasos
        # - Algunas colisiones y salidas falsas
        expected_efficiency = 0.5  # 50% de eficiencia en pasos
        expected_success_fitness = BASE_SCORE * 3 + (1 - expected_efficiency) * BASE_SCORE
        expected_success_fitness = int(expected_success_fitness * 0.90)  # 90% del esperado
        
        # ✅ Fitness MÁXIMO cuando NO encuentra salida (mejor caso sin éxito)
        # - Mejor progreso posible sin encontrar salida
        max_progress = 1.0  # 100% de mejora en distancia
        max_efficiency = 1.0  # 0% colisiones
        min_distance = 1  # Distancia mínima sin llegar
        min_false_exits = 0
        min_colissions = 0
        optimal_steps = self.num_steps * 0.3  # Usa solo 30% de pasos
        
        max_no_exit_fitness = (
            BASE_SCORE + 
            (max_progress * BASE_SCORE * 2) + 
            (max_efficiency * BASE_SCORE * 0.5) -
            (min_false_exits * 50) -
            (min_distance * 3) -
            (min_colissions * 8)
        )
        
        # ✅ El target debe estar CLARAMENTE por encima del mejor caso sin salida
        # y por debajo del peor caso con salida
        target = max(min_success_fitness, int(max_no_exit_fitness * 1.5))
        
        # Ajustar por dificultad del laberinto
        difficulty = (self.maze.wall_prob * 0.4 + 
                    self.maze.mutation_prob * 0.3 + 
                    min(self.maze.size / 100, 0.3))
        
        # En laberintos difíciles, aceptar fitness más bajo
        adjusted_target = target * (1 - difficulty * 0.2)
        
        return max(int(max_no_exit_fitness * 1.2), int(adjusted_target))

    def simulation_maze(self, cromosoma, maze):
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
                if not maze.is_valid_exit(maze.get_agent_position()):
                    find_false_exits += 1
                else:
                    #print("Salida encontrada")
                    find_exit = True
                    break
            else:
                num_colissions += 1

            maze.mutate_walls()

        agent_pos = maze.get_agent_position()
        distance_to_exit = maze.manhattan_distance(agent_pos, valid_exit_pos)

        cromosoma.set_fitness(self.fitness(num_colissions, distance_to_exit,
                                           find_false_exits, find_exit, cromosoma.get_steps_used(), initial_distance))

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

        # Fitness máximo teórico (encontrar salida sin penalizaciones)
        target_fitness = self.calculate_target_fitness() - self.calculate_target_fitness() * 0.4
        #print(target_fitness)

        best_overall = None
        best_maze_state = None
        best_fitness = -1

        for generation in range(self.max_generations):
            #print(f"\n--- Generación {generation+1} ---")

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
            #print(f"Mejor fitness: {current_fitness}")

            # Actualizar mejor global si es necesario
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_overall = best_current
                best_maze_state = copy.deepcopy(current_maze_state)  # Guardar copia del laberinto
                #print("¡Nuevo mejor global encontrado!")

            # Mostrar el laberinto del mejor de esta generación
            #print(f"\nLaberinto recorrido por el mejor cromosoma (Generación {generation+1}):")
            #current_maze_state.print_maze()
            #print(f"Posición final del agente: {current_maze_state.get_agent_position()}")
            #print(f"Salida válida: {current_maze_state.get_valid_exit()}")

            # Condición de parada si alcanzó fitness objetivo
            if current_fitness >= target_fitness:
                #print("\n✅ Solución encontrada!")
                #print(f"Número de generaciones: {generation+1}")
                return best_overall, best_maze_state

            # Generar nueva población
            self.crossing_chromosomes()

        #print(f"\n⚠️ Mejor salida encontrada en el número máximo de generaciones ({self.max_generations})")
        #if best_maze_state:
            #print("\nLaberinto final del mejor cromosoma global:")
            #best_maze_state.print_maze()
            #print(f"Posición final del agente: {best_maze_state.get_agent_position()}")
            #print(f"Salida válida: {best_maze_state.get_valid_exit()}")
            #print(f"Fitness alcanzado: {best_fitness}")
        return best_overall, best_maze_state

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