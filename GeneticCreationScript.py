import itertools
from random import random, randrange, sample
import bpy
import bmesh
import copy

reference_model = None

class Model(object):

    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = 0

        if reference_model is not None:      
            for position, reference_position in zip(self.chromosome, reference_model.vertices):
                self.fitness -= (position[0] - reference_position.co[0]) * (position[0] - reference_position.co[0]) + (position[1] - reference_position.co[1]) * (position[1] - reference_position.co[1]) + (position[2] - reference_position.co[2]) * (position[2] - reference_position.co[2])
        else:
            for position in self.chromosome:
                self.fitness += (position[0]*position[0] + position[1]*position[1] + position[2]*position[2])

            # for position in self.chromosome:
            #     distance = abs((position[0]*position[0] + position[1]*position[1] + position[2]*position[2]) - 1)
            #     if distance > 0.05:
            #         self.fitness -= distance

    def get_vertices_positions(self):
        return self.chromosome

    def uniform_crossover(self, second_model):
        first_child_positions = []
        second_child_positions = []
        for i in range(len(self.chromosome)):
            if random() < 0.5:
                first_child_positions.append(self.chromosome[i])
                second_child_positions.append(second_model.chromosome[i])
            else:
                first_child_positions.append(second_model.chromosome[i])
                second_child_positions.append(self.chromosome[i])

        return [Model(first_child_positions), Model(second_child_positions)]

    def one_point_crossover(self, second_model):
        crossover_point = randrange(len(self.chromosome))
        first_child_positions = []
        second_child_positions = []
        for i in range(len(self.chromosome)):
            if i < crossover_point:
                first_child_positions.append(self.chromosome[i])
                second_child_positions.append(second_model.chromosome[i])
            else:
                first_child_positions.append(second_model.chromosome[i])
                second_child_positions.append(self.chromosome[i])

        return [Model(first_child_positions), Model(second_child_positions)]
        
    def multi_point_crossover(self, second_model):
        crossover_points = [randrange(len(self.chromosome))]
        crossover_points.append(randrange(len(self.chromosome)))
        crossover_points.append(randrange(len(self.chromosome)))
        crossover_points.sort()
        crossover_index = 0
        first_child_positions = []
        second_child_positions = []
        for i in range(len(self.chromosome)):
            if crossover_index< len(crossover_points) and i == crossover_points[crossover_index]:
                # swap lists
                second_child_positions, first_child_positions = first_child_positions, second_child_positions
                crossover_index+=1
            first_child_positions.append(self.chromosome[i])
            second_child_positions.append(second_model.chromosome[i])

        return [Model(first_child_positions), Model(second_child_positions)]

    def crossover(self, second_model, crossover_method):
        if crossover_method == 0:
            return self.uniform_crossover(second_model)
        if crossover_method == 1:
            return self.one_point_crossover(second_model)
        if crossover_method == 2:
            return self.multi_point_crossover(second_model)
        return []

    def mutate(self):
        first_position = randrange(len(self.chromosome))
        second_position = randrange(len(self.chromosome))       
        self.chromosome[second_position], self.chromosome[first_position] = self.chromosome[first_position], self.chromosome[second_position]      
        self.calculate_fitness()

def insert_keyframe(fcurves, frame, values):
    for fcu, val in zip(fcurves, values):
        fcu.keyframe_points.insert(frame, val, options={'FAST'})

def create_reference_sphere(u, v):
    mesh = bpy.data.meshes.new('ReferenceSphere')
    sphere = bpy.data.objects.new("ReferenceSphere", mesh)
    bpy.context.collection.objects.link(sphere)
    # construct the bmesh sphere and assign it to the blender mesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1)
    bm.to_mesh(mesh)
    bm.free()
    return mesh

def initialize_population(population_size, chromosome_size, space_size, space_bounds):
    population = []
    for _ in range (population_size):
        modelVertices = []
        for _ in range(0, chromosome_size):
            x = random() * space_size + space_bounds[0]
            y = random() * space_size + space_bounds[0]
            z = random() * space_size + space_bounds[0]
            modelVertices.append([x, y, z])
        population.append(Model(modelVertices))
    return population

def create_mesh(vertices):
    if reference_model is not None:
        edges = [e.vertices for e in reference_model.edges]
        faces = [f.vertices for f in reference_model.polygons]
    else:
        edges = []
        faces = []
    new_mesh = bpy.data.meshes.new('OptimizedMesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    # make object from mesh
    new_object = bpy.data.objects.new('OptimizedMesh', new_mesh)
    # make collection
    new_collection = bpy.data.collections.new('OptimizationResults')
    bpy.context.scene.collection.children.link(new_collection)
    # add object to scene collection
    new_collection.objects.link(new_object)
    return new_mesh

def create_animation(new_mesh, iterations_count, animation_positions):
        action = bpy.data.actions.new("MeshAnimation")
        new_mesh.animation_data_create()
        new_mesh.animation_data.action = action

        data_path = "vertices[%d].co"
        frames = range(0, iterations_count * 5, 5)

        for v in new_mesh.vertices:
            fcurves = [action.fcurves.new(data_path % v.index, index =  i) for i in range(3)]

            for t, animation_position in zip(frames, animation_positions):
                insert_keyframe(fcurves, t, animation_position[v.index])  

def get_elite_from_sorted_popuation(population, count):
    if count == 1:
        return [population[0]]
    if count < 1:
        return []
    elite = population[:count - 1]
    elite.append(copy.deepcopy(population[0]))
    elite[-1].mutate()
    return elite

def select_parent_turnament(population, count, is_maximized):
    turnament_selection = sample(population, count)
    turnament_selection.sort(key=lambda m: m.fitness, reverse=is_maximized)
    return turnament_selection[0]

def select_parent_rulette(population, partial_sums, fitness_sum):
    parent_threshold = randrange(int(fitness_sum) + 1)
    parent_index = next(i for i, val in enumerate(partial_sums) if val > parent_threshold)
    return population[parent_index]

def is_model_better(new_model, old_model, is_maximized):
    if old_model is None:
        return True
    if is_maximized and new_model.fitness > old_model.fitness:
        return True
    elif not is_maximized and new_model.fitness < old_model.fitness:
        return True
    return False

def main():
    # genetic algorithm parameters
    population_size = 6000
    mutation_probability = 0.2
    max_generations = 150
    turnament_count = 3
    max_generations_without_improvement = 5
    use_elitism = False
    elite_count = 2
    selection_method = 1 # 0: rulette selection, 1: turnament selection
    crossover_method = 2 # 0: uniform crossover, 1: one point crossover, 2: multi point crossover

    # mode parameters
    use_points_mode = False # in this mode, the positions will be optimized to be as far away from the center of the space as possible
    chromosome_size = 100 # this value is used only in points mode
    u_sphere = 8
    v_sphere = 7

    # program parametrs
    animate_results = True
    space_bounds = [-1, 1] # all vertices positions will be created inside this space in each axis (X, Y, Z)

    # calculate mode specific model
    if not use_points_mode:
        selected_obj = bpy.context.selected_objects[0]
        global reference_model
        if selected_obj is not None:
            reference_model = selected_obj.data
        else:
            # create blender uv sphere
            reference_model = create_reference_sphere(u_sphere, v_sphere)
        chromosome_size = len(reference_model.vertices)
        
    # initilize population
    space_size = space_bounds[1] - space_bounds[0]
    population = initialize_population(population_size, chromosome_size, space_size, space_bounds)

    #program variables
    is_maximized = True
    generations_without_improvement_count = 0
    best_model = None
    animation_positions = []
    last_fitness = 0
    generation = 0

    while generation <= max_generations:
        offspring = []
        if selection_method == 0:
            partial_sums = list(itertools.accumulate(model.fitness for model in population))
            fitness_sum = partial_sums[-1]
            print(fitness_sum)

        for _ in range(int(population_size / 2)):
            # selection
            if selection_method == 0:
                first_parent = select_parent_rulette(population, partial_sums, fitness_sum)
                second_parent = select_parent_rulette(population, partial_sums, fitness_sum)
            elif selection_method == 1:
                first_parent = select_parent_turnament(population, turnament_count, is_maximized)
                second_parent = select_parent_turnament(population, turnament_count, is_maximized)

            # crossover
            offspring.extend(first_parent.crossover(second_parent, crossover_method))

            # mutation
            if random() < mutation_probability:
                offspring[-1].mutate()
            if random() < mutation_probability:
                offspring[-2].mutate()

        population.sort(key=lambda m: m.fitness, reverse=is_maximized)

        # elitism
        if use_elitism:
            offspring.extend(get_elite_from_sorted_popuation(population, elite_count))
 
        print('Generation: ', generation, ' Fitness: ', population[0].fitness)

        if is_model_better(population[0], best_model, is_maximized):
            best_model = population[0]

        if population[0].fitness == last_fitness:
            generations_without_improvement_count += 1
        else:
            last_fitness = population[0].fitness
            generations_without_improvement_count = 0
        if generations_without_improvement_count == max_generations_without_improvement:
            break

        if animate_results:
            animation_positions.append(population[0].get_vertices_positions())

        population = offspring
        generation += 1

    population.sort(key=lambda m: m.fitness, reverse=is_maximized)
    if is_model_better(population[0], best_model, is_maximized):
        best_model = population[0]

    # create mesh
    positions = best_model.get_vertices_positions()
    new_mesh = create_mesh(positions)

    if animate_results:
        animation_positions.append(positions)
        create_animation(new_mesh, generation, animation_positions)

if __name__ == '__main__':
    main()
