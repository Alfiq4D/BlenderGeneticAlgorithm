from random import random, randrange, sample
import bpy
import bmesh
import copy

reference_model = None

class Parameters(object):
    # genetic algorithm parameters
    population_size = 6000
    mutation_probability = 0.2
    max_generations = 150
    turnament_count = 3
    max_generations_without_improvement = 5
    use_elitism = False
    elite_count = 2
    crossover_method = 2 # 0: uniform crossover, 1: one point crossover, 2: multi point crossover
    # mode parameters
    mode = 0 # 0: Points Mode, 1: Sphere mode, 2: Custom object mode
    chromosome_size = 100 # used only in points mode
    u_sphere = 8 # used only in sphere mode
    v_sphere = 7 # used only in sphere mode
    # program parametrs
    animate_results = True
    space_bounds = [-1, 1] # all vertices positions will be randomly created inside this compartment on each axis (X, Y, Z)

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

class GeneticOptimizer(object):
    is_maximized = True

    def __init__(self, population_size, chromosome_size, space_size, space_bounds):
        self.generation = 0
        self.best_model = None
        self.population = []
        for _ in range (population_size):
            modelVertices = []
            for _ in range(0, chromosome_size):
                x = random() * space_size + space_bounds[0]
                y = random() * space_size + space_bounds[0]
                z = random() * space_size + space_bounds[0]
                modelVertices.append([x, y, z])
            self.population.append(Model(modelVertices))

    def get_elite_from_sorted_popuation(self, count):
        if count == 1:
            return [self.population[0]]
        if count < 1:
            return []
        elite = self.population[:count - 1]
        elite.append(copy.deepcopy(self.population[0]))
        elite[-1].mutate()
        return elite

    def select_parent_turnament(self, count, is_maximized):
        turnament_selection = sample(self.population, count)
        turnament_selection.sort(key=lambda m: m.fitness, reverse=is_maximized)
        return turnament_selection[0]

    def update_best_model(self, new_model):
        if self.best_model is None:
            self.best_model = new_model
        if self.is_maximized and new_model.fitness > self.best_model.fitness:
            self.best_model = new_model
        elif not self.is_maximized and new_model.fitness < self.best_model.fitness:
            self.best_model = new_model

    def optimize(self, context):
        best_positions = []
        last_fitness = 0
        generations_without_improvement_count = 0

        while self.generation <= context.max_generations:      
            self.population.sort(key=lambda m: m.fitness, reverse=self.is_maximized)
            self.update_best_model(self.population[0])
            print('Generation: ', self.generation, ' Fitness: ', self.population[0].fitness)

            # check break condition
            if self.population[0].fitness == last_fitness:
                generations_without_improvement_count += 1
            else:
                last_fitness = self.population[0].fitness
                generations_without_improvement_count = 0
            if generations_without_improvement_count == context.max_generations_without_improvement:
                break

            if context.animate_results:
                best_positions.append(self.population[0].get_vertices_positions())

            offspring = []
            for _ in range(int(context.population_size / 2)):
                # selection
                first_parent = self.select_parent_turnament(context.turnament_count, self.is_maximized)
                second_parent = self.select_parent_turnament(context.turnament_count, self.is_maximized)

                # crossover
                offspring.extend(first_parent.crossover(second_parent, context.crossover_method))

                # mutation
                if random() < context.mutation_probability:
                    offspring[-1].mutate()
                if random() < context.mutation_probability:
                    offspring[-2].mutate()

            # elitism
            if context.use_elitism:
                offspring.extend(self.get_elite_from_sorted_popuation(context.elite_count))

            self.population = offspring
            self.generation += 1

        self.population.sort(key=lambda m: m.fitness, reverse=self.is_maximized)
        self.update_best_model(self.population[0])
        if context.animate_results:
            best_positions.append(self.best_model.get_vertices_positions())

        return self.best_model, best_positions

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

def main():
    parameters = Parameters()  
    # calculate mode specific model
    if parameters.mode == "2":
        global reference_model
        if bpy.context.selected_objects and bpy.context.selected_objects[0] is not None and bpy.context.selected_objects[0].type == 'MESH':
            reference_model = bpy.context.selected_objects[0].data
            parameters.chromosome_size = len(reference_model.vertices)
        else:
            return
    elif parameters.mode == "1":
        # create blender uv sphere
        reference_model = create_reference_sphere(parameters.u_sphere, parameters.v_sphere)
        parameters.chromosome_size = len(reference_model.vertices)
    
    # perform optimization
    space_size = parameters.space_bounds[1] - parameters.space_bounds[0]
    optimizer = GeneticOptimizer(parameters.population_size, parameters.chromosome_size, space_size, parameters.space_bounds)
    best_model, animation_positions = optimizer.optimize(context=parameters)
    
    # create mesh
    new_mesh = create_mesh(best_model.get_vertices_positions())
    if parameters.animate_results:
        create_animation(new_mesh, optimizer.generation, animation_positions)

if __name__ == '__main__':
    main()
