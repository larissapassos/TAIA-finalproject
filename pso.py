import numpy as np
from random import random

class Particle:
    def __init__(self, position, velocity, fitness):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.p_best = position
        self.p_best_fitness = fitness

    def __cmp__(self, other):
        return self.fitness - other.fitness

class Pso:
    def __init__(self, iterations=10000, pop_size=5, dimensions=2, w=0.5, c1=0.5, c2=1, limiar=0.9):
        self.iterations = iterations
        self.population_size = pop_size
        self.dimensions = dimensions
        self.weight = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = 0
        self.population = []
        self.g_best = 0
        self.g_best_fitness = 0
        self.fitness_function = None
        self.limiar = limiar

    def get_velocity(self, particle):
        inertia = self.weight * particle.velocity
        cognitive = self.c1 * random() * (particle.p_best - particle.position)
        social = self.c2 * random() * (self.g_best - particle.position)
        particle.velocity = inertia + cognitive + social

    def get_position(self, particle):
        particle.position = particle.position + particle.velocity

    def get_fitness(self, particle):
        # call extern function to calculate fitness
        particle.fitness = self.fitness_function(particle.position)

        if particle.fitness > self.g_best_fitness:
            self.g_best = particle.position
            self.g_best_fitness = particle.fitness
        if particle.fitness > particle.p_best_fitness:
            particle.p_best = particle.position
            particle.p_best_fitness = particle.fitness


    def get_g_best(self):
        sorted_pop = sorted(self.population, reverse = True)
        best = sorted_pop[0]
        self.g_best = best.position
        self.g_best_fitness = best.fitness

    def init_population(self, population):
        for solution in population:
            initial_velocity = np.random.random(self.dimensions)
            fitness = self.fitness_function(solution)
            particle = Particle(solution, initial_velocity, fitness)
            self.population.append(particle)
        self.get_g_best()

    def run(self, population, fitness_function, verbose=0):
        self.fitness_function = fitness_function
        self.verbose = verbose
        self.init_population(population)
        solution = {}
        rounds = 0
        if self.verbose == 1:
            print "iteration: %d" % rounds
            print self.g_best, self.g_best_fitness
        while rounds < self.iterations and self.g_best_fitness < self.limiar:
            for p in self.population:
                self.get_velocity(p)
                self.get_position(p)
                self.get_fitness(p)
            rounds += 1
            if self.verbose == 1:
                print "iteration: %d" % rounds
                print self.g_best, self.g_best_fitness
        solution['iterations'] = rounds
        solution['g_best'] = self.g_best
        solution['g_best_fitness'] = self.g_best_fitness
        return solution

# Testing methods

# def test_fitness(position):
#     return position[0] ** 2 + position[1] ** 2

# def get_pop():
#     pop = []
#     for i in xrange(5):
#         pop.append(np.random.randint(11, size=2))
#     return pop

# population = get_pop()

# pso = Pso(iterations = 10, limiar = 200)
# # pso.fitness_function = test_fitness
# # pso.init_population(population)

# # print "Particles:"
# # for p in pso.population:
# #     print p.position, p.velocity, p.fitness, p.p_best, p.p_best_fitness
# #     pso.get_velocity(p)
# #     pso.get_position(p)
# #     pso.get_fitness(p)
# #     print p.position, p.velocity, p.fitness, p.p_best, p.p_best_fitness
# #     print pso.g_best, pso.g_best_fitness
# solution = pso.run(population, test_fitness, verbose=1)
# print solution