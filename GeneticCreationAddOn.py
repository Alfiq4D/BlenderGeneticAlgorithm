from random import random, randrange, sample
import bpy
import bmesh
import copy

bl_info = {
    "name": "Genetic Optimizer",
    "blender": (3, 30, 0),
    "category": "Object",
}

reference_model = None

class Model(object):

    def __init__(self, chromosome):
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
        if crossover_method == "U":
            return self.uniform_crossover(second_model)
        if crossover_method == "O":
            return self.one_point_crossover(second_model)
        if crossover_method == "M":
            return self.multi_point_crossover(second_model)
        return []

    def mutate(self):
        first_position = randrange(len(self.chromosome))
        second_position = randrange(len(self.chromosome))       
        self.chromosome[second_position], self.chromosome[first_position] = self.chromosome[first_position], self.chromosome[second_position]      
        self.calculate_fitness()

class GeneticOptimizer:
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

        while self.generation <= context.scene.max_generations:      
            self.population.sort(key=lambda m: m.fitness, reverse=self.is_maximized)
            self.update_best_model(self.population[0])
            print('Generation: ', self.generation, ' Fitness: ', self.population[0].fitness)

            # check break condition
            if self.population[0].fitness == last_fitness:
                generations_without_improvement_count += 1
            else:
                last_fitness = self.population[0].fitness
                generations_without_improvement_count = 0
            if generations_without_improvement_count == context.scene.max_generations_without_improvement:
                break

            if context.scene.animate_results:
                best_positions.append(self.population[0].get_vertices_positions())

            offspring = []
            for _ in range(int(context.scene.population_size / 2)):
                # selection
                first_parent = self.select_parent_turnament(context.scene.turnament_count, self.is_maximized)
                second_parent = self.select_parent_turnament(context.scene.turnament_count, self.is_maximized)

                # crossover
                offspring.extend(first_parent.crossover(second_parent, context.scene.crossover_method))

                # mutation
                if random() < context.scene.mutation_probability:
                    offspring[-1].mutate()
                if random() < context.scene.mutation_probability:
                    offspring[-2].mutate()

            # elitism
            if context.scene.use_elitism:
                offspring.extend(self.get_elite_from_sorted_popuation(context.scene.elite_count))

            self.population = offspring
            self.generation += 1

        self.population.sort(key=lambda m: m.fitness, reverse=self.is_maximized)
        self.update_best_model(self.population[0])
        if context.scene.animate_results:
            best_positions.append(self.best_model.get_vertices_positions())

        return self.best_model, best_positions

class GeneticOperator(bpy.types.Operator):
    """Genetic Optimization"""
    bl_idname = "genetic.optimization"
    bl_label = "Genetic Optimization"
    bl_options = {'REGISTER', 'UNDO'}

    def insert_keyframe(self, fcurves, frame, values):
        for fcu, val in zip(fcurves, values):
            fcu.keyframe_points.insert(frame, val, options={'FAST'})

    def create_reference_sphere(self, u, v):
        mesh = bpy.data.meshes.new('ReferenceSphere')
        sphere = bpy.data.objects.new("ReferenceSphere", mesh)
        bpy.context.collection.objects.link(sphere)
        # construct the bmesh sphere and assign it to the blender mesh
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    def create_mesh(self, vertices):
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

    def create_animation(self, new_mesh, iterations_count, animation_positions):
        action = bpy.data.actions.new("MeshAnimation")
        new_mesh.animation_data_create()
        new_mesh.animation_data.action = action

        data_path = "vertices[%d].co"
        frames = range(0, iterations_count * 5, 5)

        for v in new_mesh.vertices:
            fcurves = [action.fcurves.new(data_path % v.index, index =  i) for i in range(3)]

            for t, animation_position in zip(frames, animation_positions):
                self.insert_keyframe(fcurves, t, animation_position[v.index])

    def execute(self, context):
        # calculate mode specific model
        if context.scene.mode == "C":
            global reference_model
            if bpy.context.selected_objects and bpy.context.selected_objects[0] is not None and bpy.context.selected_objects[0].type == 'MESH':
                reference_model = bpy.context.selected_objects[0].data
                context.scene.chromosome_size = len(reference_model.vertices)
            else:
                return
        elif context.scene.mode == "S":
            # create blender uv sphere
            reference_model = self.create_reference_sphere(context.scene.u_sphere, context.scene.v_sphere)
            context.scene.chromosome_size = len(reference_model.vertices)
        
        # perform optimization
        space_size = context.scene.space_bounds[1] - context.scene.space_bounds[0]
        optimizer = GeneticOptimizer(context.scene.population_size, context.scene.chromosome_size, space_size, context.scene.space_bounds)
        best_model, animation_positions = optimizer.optimize(context=context)
        
        # create mesh
        new_mesh = self.create_mesh(best_model.get_vertices_positions())
        if context.scene.animate_results:
            self.create_animation(new_mesh, optimizer.generation, animation_positions)

        return {'FINISHED'}

class ParametersPanel(bpy.types.Panel):
    """Create Optimization Panel"""
    bl_label = "Genetic Optimization"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Genetic Optimization"

    def draw(self, context):
        column = self.layout.column()
        
        column.prop(context.scene, "population_size")
        column.prop(context.scene, "mutation_probability")
        column.prop(context.scene, "max_generations")
        column.prop(context.scene, "turnament_count")
        column.prop(context.scene, "max_generations_without_improvement")
        column.prop(context.scene, "use_elitism")
        column.prop(context.scene, "elite_count")
        column.prop(context.scene, "crossover_method")
        column.prop(context.scene, "space_bounds")
        column.separator()
        column.prop(context.scene, "mode")
        column.label(text="Points mode:")
        column.prop(context.scene, "chromosome_size")
        column.label(text="Sphere mode:")
        column.prop(context.scene, "u_sphere")
        column.prop(context.scene, "v_sphere")
        column.separator()
        column.prop(context.scene, "animate_results")
        column.separator()
        column.operator(GeneticOperator.bl_idname, text="Run optimization")

classes = [
    ParametersPanel,
    GeneticOperator,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    # genetic algorithm parameters
    bpy.types.Scene.population_size = bpy.props.IntProperty(name="Population size", default=6000, min=1, max=50000)
    bpy.types.Scene.mutation_probability = bpy.props.FloatProperty(name="Mutation probability", default=0.2, min=0, max=1)
    bpy.types.Scene.max_generations = bpy.props.IntProperty(name="Max generations", default=150, min=1, max=1000)
    bpy.types.Scene.turnament_count = bpy.props.IntProperty(name="Turnament count", default=3, min=1, max=20)
    bpy.types.Scene.max_generations_without_improvement = bpy.props.IntProperty(name="Max generations without improvement", default=5, min=1, max=10)
    bpy.types.Scene.use_elitism = bpy.props.BoolProperty(name="Use elitism", default=False)
    bpy.types.Scene.elite_count = bpy.props.IntProperty(name="Elite count", default=2, min=1, max=5)
    bpy.types.Scene.crossover_method = bpy.props.EnumProperty(items=[("U", "Uniform crossover", ""), ("O", "One point crossover", ""), ("M", "Multi point crossover", ""),], name="Crossover method", default="M") # 0: uniform crossover, 1: one point crossover, 2: multi point crossover
    # mode parameters
    bpy.types.Scene.mode = bpy.props.EnumProperty(items=[("P", "Points Mode", ""), ("S", "Sphere mode", ""), ("C", "Custom object mode", ""),], name="Mode", default="C")
    bpy.types.Scene.chromosome_size = bpy.props.IntProperty(name="Chromosome size", default=100, min=1, max=1000) # this value is used only in points mode
    bpy.types.Scene.u_sphere = bpy.props.IntProperty(name="U sphere", default=8, min=2, max=20)
    bpy.types.Scene.v_sphere = bpy.props.IntProperty(name="V sphere", default=7, min=2, max=20)
    # program parametrs
    bpy.types.Scene.animate_results = bpy.props.BoolProperty(name="Animate results", default=True)
    bpy.types.Scene.space_bounds = bpy.props.FloatVectorProperty(name="Space bounds", default=(-1.0, 1.0), min=-2.0, max=2.0, size=2) # all vertices positions will be created inside this space in each axis (X, Y, Z)

def unregister():
    del bpy.types.Scene.population_size
    del bpy.types.Scene.mutation_probability
    del bpy.types.Scene.max_generations
    del bpy.types.Scene.turnament_count
    del bpy.types.Scene.max_generations_without_improvement
    del bpy.types.Scene.use_elitism
    del bpy.types.Scene.elite_count
    del bpy.types.Scene.crossover_method
    del bpy.types.Scene.mode
    del bpy.types.Scene.chromosome_size
    del bpy.types.Scene.u_sphere
    del bpy.types.Scene.v_sphere
    del bpy.types.Scene.animate_results
    del bpy.types.Scene.space_bounds   
    for c in classes:
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
    