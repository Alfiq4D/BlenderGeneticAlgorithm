from math import sqrt
from random import random, randrange, sample
import bpy
import bmesh
import copy

reference_model = None
reference_edges = []

class Model(object):

    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = 0
        # for position in self.chromosome:
        #     self.fitness += (position[0]*position[0] + position[1]*position[1] + position[2]*position[2])

        for position, referencePosition in zip(self.chromosome, reference_model.vertices):
            self.fitness -= (position[0] - referencePosition.co[0]) * (position[0] - referencePosition.co[0]) + (position[1] - referencePosition.co[1]) * (position[1] - referencePosition.co[1]) + (position[2] - referencePosition.co[2]) * (position[2] - referencePosition.co[2])

        # for position in self.chromosome:
        #     distance = abs((position[0]*position[0] + position[1]*position[1] + position[2]*position[2]) - 1)
        #     if distance > 0.05:
        #         self.fitness -= distance

    def get_vertices_positions(self):
        return self.chromosome

    def crossover(self, secondModel):
        firstChildPositions = []
        secondChildPositions = []
        for i in range(len(self.chromosome)):
            if random() < 0.5:
                firstChildPositions.append(self.chromosome[i])
                secondChildPositions.append(secondModel.chromosome[i])
            else:
                firstChildPositions.append(secondModel.chromosome[i])
                secondChildPositions.append(self.chromosome[i])

        return [Model(firstChildPositions), Model(secondChildPositions)]

    def one_point_crossover(self, second_model):
        crossover_point = randrange(len(self.chromosome))
        firstChildPositions = []
        secondChildPositions = []
        for i in range(len(self.chromosome)):
            if i < crossover_point:
                firstChildPositions.append(self.chromosome[i])
                secondChildPositions.append(second_model.chromosome[i])
            else:
                firstChildPositions.append(second_model.chromosome[i])
                secondChildPositions.append(self.chromosome[i])

        return [Model(firstChildPositions), Model(secondChildPositions)]
        
    def multi_point_crossover(self, second_model):
        crossover_points = [randrange(len(self.chromosome))]
        crossover_points.append(randrange(len(self.chromosome)))
        crossover_points.append(randrange(len(self.chromosome)))
        crossover_points.sort()
        crossover_index = 0
        firstChildPositions = []
        secondChildPositions = []
        for i in range(len(self.chromosome)):
            if crossover_index< len(crossover_points) and i == crossover_points[crossover_index]:
                # swap lists
                secondChildPositions, firstChildPositions = firstChildPositions, secondChildPositions
                crossover_index+=1
            firstChildPositions.append(self.chromosome[i])
            secondChildPositions.append(second_model.chromosome[i])

        return [Model(firstChildPositions), Model(secondChildPositions)]

    def mutate(self):
        firstPosition = randrange(len(self.chromosome))
        secondPosition = randrange(len(self.chromosome))       
        self.chromosome[secondPosition], self.chromosome[firstPosition] = self.chromosome[firstPosition], self.chromosome[secondPosition]      
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

    global reference_model
    reference_model = mesh

    global reference_edges
    reference_edges = [e.vertices for e in reference_model.edges]

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
    edges = []
    edges = [e.vertices for e in reference_model.edges]
    faces = []
    faces = [f.vertices for f in reference_model.polygons]
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

def select_parent_turnament(population, count):
    turnament_selection = sample(population, count)
    turnament_selection.sort(key=lambda m: m.fitness, reverse=True)
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
    # parameters
    u_sphere = 8
    v_sphere = 7
    # create blender uv sphere
    create_reference_sphere(u_sphere, v_sphere)

    selected_obj = bpy.context.selected_objects[0]
    if selected_obj is not None:
        global reference_model
        reference_model = selected_obj.data

    turnament_count = 3
    elite_count = 2
    mutation_probability = 0.2
    population_size = 6000
    # chromosome_size = 100
    # chromosome_size = u_sphere * (v_sphere - 1) + 2
    chromosome_size = len(reference_model.vertices)
    max_generations = 150
    max_generations_without_improvement = 5
    generations_without_improvement_count = 0
    best_model = None
    animate = True
    animation_positions = []
    space_bounds = [-1, 1] # all vertices positions will be created inside this space
    space_size = space_bounds[1] - space_bounds[0]
    is_maximized = True



    # initilize population
    population = initialize_population(population_size, chromosome_size, space_size, space_bounds)

    last_fitness = 0
    generation = 0 
    while generation <= max_generations:
        # partial_sums = list(itertools.accumulate(model.fitness for model in population))
        # fitness_sum = partial_sums[-1]
        offspring = []

        for _ in range(int(population_size / 2)):

            # selection
            # first_parent = select_parent_rulette(population, partial_sums, fitness_sum)
            first_parent = select_parent_turnament(population, turnament_count)
            # second_parent = select_parent_rulette(population, partial_sums, fitness_sum)
            second_parent = select_parent_turnament(population, turnament_count)

            # crossover
            offspring.extend(first_parent.crossover(second_parent))

            # mutation
            if random() < mutation_probability:
                offspring[-1].mutate()
            if random() < mutation_probability:
                offspring[-2].mutate()

        population.sort(key=lambda m: m.fitness, reverse=True)

        # elitism
        # offspring.extend(get_elite_from_sorted_popuation(population, elite_count))
 
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

        if animate:
            animation_positions.append(population[0].get_vertices_positions())

        population = offspring
        generation += 1

    population.sort(key=lambda m: m.fitness, reverse=True)
    if is_model_better(population[0], best_model, is_maximized):
        best_model = population[0]

    # make mesh
    positions = best_model.get_vertices_positions()
    new_mesh = create_mesh(positions)

    if animate:
        animation_positions.append(positions)
        create_animation(new_mesh, generation, animation_positions)

if __name__ == '__main__':
    main()
