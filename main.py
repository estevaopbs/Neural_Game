import copy
from bisect import bisect_left
#import numba
#from numba import cuda
#import pandas as pd
import time
import pygame
from pygame.locals import *
from sys import exit
#import cupy as np
import numpy as np
from Neural import *
#import code
# code.interact(local=locals)


def draw_tiles(surface, coordinates, inverted=False):
    i = 0
    if inverted:
        i = 1
    x0 = coordinates[0][0]
    y0 = coordinates[0][1]
    x = coordinates[1][0]
    y = coordinates[1][1]
    x_tiles = int((x - x0) / 50)
    y_tiles = int((y - y0) / 50)
    for n, tile in enumerate(range(y_tiles)):
        for m, _tile in enumerate(range(x_tiles)):
            if not m + n + i & 1:
                pygame.draw.rect(surface, (255, 255, 255),
                                 ((m * 50 + x0, n * 50 + y0), (50, 50)))
            else:
                pygame.draw.rect(surface, (229, 228, 255),
                                 ((m * 50 + x0, n * 50 + y0), (50, 50)))


def level_1():
    def draw_background():
        map_surf.fill(background_color)
        draw_tiles(map_surf, ((211, 161), (711, 361)))
        draw_tiles(map_surf, ((161, 361), (261, 411)), True)
        draw_tiles(map_surf, ((661, 111), (761, 161)))
        pygame.draw.rect(map_surf, color=checkpoint_color,
                         rect=((11, 111), (150, 300)))
        pygame.draw.rect(map_surf, color=checkpoint_color,
                         rect=((761, 111), (150, 300)))
        individuals_surf.fill([255, 255, 255])

    def draw_walls():
        pygame.draw.lines(map_surf,
                          (0, 0, 0),
                          points=[(200, 155), (660, 155), (655, 155), (655, 100), (655, 105), (921, 105), (916, 105),
                                  (916, 421), (916, 416), (751, 416), (756,
                                                                       416), (756, 161), (756, 166), (711, 166),
                                  (716, 166), (716, 371), (716, 366), (261,
                                                                       366), (266, 366), (266, 421), (266, 416),
                                  (0, 416), (5, 416), (5, 100), (5,
                                                                 105), (171, 105), (166, 105), (166, 360),
                                  (166, 355), (210, 355), (205, 355), (205, 160)],
                          width=11,
                          closed=False)

    def vel_function(obstacle_):
        if obstacle_.pos[0] <= 224.5:
            obstacle_.pos[0] = 224.5
            obstacle_.vel = - obstacle_.vel
        elif obstacle_.pos[0] >= 698.5:
            obstacle_.pos[0] = 698.5
            obstacle_.vel = - obstacle_.vel

    initial_pos = np.array([86, 236])
    obstacles_ = [Obstacle(np.array([224.5, 186.5]), np.array([-11, 0]), vel_function),
                  Obstacle(np.array([224.5, 286.5]),
                           np.array([-11, 0]), vel_function),
                  Obstacle(np.array([698.5, 336.5]),
                           np.array([11, 0]), vel_function),
                  Obstacle(np.array([698.5, 236.5]), np.array([11, 0]), vel_function)]
    return Run_Level(draw_background, draw_walls, obstacles_), initial_pos


class Run_Level:
    def __init__(self, draw_background, draw_walls, _obstacles):
        self._obstacles = _obstacles
        self.draw_background = draw_background
        self.draw_walls = draw_walls

    def obstacles(self):
        obstacles_surf.fill([255, 255, 255])
        for obstacle in self._obstacles:
            obstacle.draw_collision()

    def obstacles_color(self):
        for obstacle in self._obstacles:
            obstacle.draw_color()


class Individual:
    def __init__(self, pos, vel, vision_rays, neural_network, draw_ray_casting=False,
                 output_function=Neuron.binary, color=None):
        self.pos = pos
        self.vel = vel
        self.vision_rays = vision_rays
        self.neural_network = neural_network
        if color is None:
            self.color = [random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)]
        else:
            self.color = color
        self.draw_ray_casting = draw_ray_casting
        self.fitness = -1
        self.alive = True
        self.output_function = output_function
        self.sensors_values = self._get_sensor_values(
            fit_array, game_surf_collision_array)
        self.age = 0
        self.higher_fitness_time = None
        self.higher_fitness = -1
        self.draw()

    def get_fitness(self):
        self.fitness = self.get_position_fitness() ** 2

    def blit(self):
        if self.alive:
            self.draw()
        else:
            self.draw_corpse()
        screen.blit(individuals_surf, self.pos + np.array([483, 263]))

    def get_higher_fitness(self):
        fitness = self.get_position_fitness()
        if fitness > self.higher_fitness:
            self.higher_fitness = fitness
            self.higher_fitness_time = time.time() - start_time

    def draw_corpse(self):
        individuals_surf.set_alpha(50)
        pygame.draw.rect(individuals_surf, (0, 0, 0), ((0, 0), (35, 35)))
        pygame.draw.rect(individuals_surf, self.color, ((5, 5), (25, 25)))

    def die(self):
        self.alive = False
        self.get_fitness()

    def get_position_fitness(self):
        return 976 - fit_array[self.pos[0]][self.pos[1]]

    def fit_sensor(self):
        return (np.array([
            fit_array[self.pos[0] + self.vel][self.pos[1]],
            fit_array[self.pos[0] + self.vel][self.pos[1] + self.vel],
            fit_array[self.pos[0]][self.pos[1] + self.vel],
            fit_array[self.pos[0] - self.vel][self.pos[1] + self.vel],
            fit_array[self.pos[0] - self.vel][self.pos[1]],
            fit_array[self.pos[0] - self.vel][self.pos[1] - self.vel],
            fit_array[self.pos[0]][self.pos[1] - self.vel],
            fit_array[self.pos[0] + self.vel][self.pos[1] - self.vel]]))

    def vision(self, map_array, pixel_array):
        vision = np.zeros([self.vision_rays])
        objects_in_sight = np.zeros([2 * self.vision_rays])
        colors = np.zeros([self.vision_rays, 3])
        coordinates = np.zeros([self.vision_rays, 2], dtype=int)
        d_theta = 2 * np.pi / self.vision_rays
        theta = 0
        for n in range(self.vision_rays):
            m = 0
            while True:
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                px = int(round(m * cos_theta + self.pos[0], 0))
                py = int(round(m * sin_theta + self.pos[1], 0))
                if pixel_array[px][py] == 0:
                    o = 1
                    is_wall = map_array[px][py] == -1
                    colors[n] = self.color if is_wall else np.array(
                        [255, 255, 255]) - self.color
                    if colors[n][0] == 0:
                        colors[n][0] += 1
                    if not is_wall:
                        objects_in_sight[n] = 1  # obstacle
                    while True:
                        px = int(round((m - o) * cos_theta + self.pos[0], 0))
                        py = int(round((m - o) * sin_theta + self.pos[1], 0))
                        if not pixel_array[px, py] == 0:
                            vision[n] = (
                                np.sqrt(((m - o + 1) * cos_theta) ** 2 + ((m - o + 1) * sin_theta) ** 2))
                            coordinates[n] = np.array([px, py])
                            theta += d_theta
                            break
                        o += 1
                    break
                m += 10

        if self.draw_ray_casting:
            Individual._draw_ray_casting(self, coordinates, colors)
        return [vision, objects_in_sight]

    def _get_obstacle_collision(self, pixel_array):
        collision = np.array([pixel_array[self.pos[0] - 17][self.pos[1]] == 0,  # left
                              # left-down
                              pixel_array[self.pos[0] - \
                                          17][self.pos[1] - 17] == 0,
                              # left-down
                              pixel_array[self.pos[0] - \
                                          17][self.pos[1] + 17] == 0,
                              pixel_array[self.pos[0] + \
                                          17][self.pos[1]] == 0,  # right
                              # right-down
                              pixel_array[self.pos[0] + \
                                          17][self.pos[1] + 17] == 0,
                              # right-up
                              pixel_array[self.pos[0] + \
                                          17][self.pos[1] - 17] == 0,
                              pixel_array[self.pos[0]
                                          ][self.pos[1] - 17] == 0,  # up
                              pixel_array[self.pos[0]
                                          ][self.pos[1] + 17] == 0  # down
                              ])

        if collision.any():
            self.die()

    def run(self, walls_collision_array, _obstacles_collision_array, game_surface_collision_array):
        self.move(self._get_movement(), walls_collision_array)
        self.get_higher_fitness()
        self.sensors_values = self._get_sensor_values(
            walls_collision_array, game_surface_collision_array)
        self._get_obstacle_collision(_obstacles_collision_array)

    def _get_movement(self):
        movement = [self.output_function(
            m) for m in self.neural_network.evaluate(self.sensors_values)]
        return movement

    def _get_sensor_values_alternative(self, wall_collision_array, game_surface_collision_array):
        vision = self.vision(wall_collision_array,
                             game_surface_collision_array)
        inputs = np.zeros([2 * self.vision_rays + 4])
        for sensor_index in range(self.vision_rays):
            if vision[1][sensor_index] == 0:
                inputs[sensor_index] = vision[0][sensor_index]
            elif vision[1][sensor_index] == 1:
                inputs[sensor_index + self.vision_rays] = vision[0][sensor_index]
        fit_sensor = self.fit_sensor()
        for fit_index in range(4):
            inputs[fit_index + 2 * self.vision_rays] = fit_sensor[fit_index] - \
                fit_sensor[fit_index + 4]
        return inputs

    def _get_sensor_values(self, wall_collision_array, game_surface_collision_array):
        vision = self.vision(wall_collision_array,
                             game_surface_collision_array)
        inputs = np.zeros([2 * self.vision_rays + 8])
        for sensor_index in range(self.vision_rays):
            if vision[1][sensor_index] == 0:
                inputs[sensor_index] = vision[0][sensor_index]
            elif vision[1][sensor_index] == 1:
                inputs[sensor_index + self.vision_rays] = vision[0][sensor_index]
        fit_sensor = self.fit_sensor()
        for fit_index in range(8):
            inputs[fit_index + 2 * self.vision_rays] = fit_sensor[fit_index]
        return inputs

    def move(self, movement, pixel_array):
        if movement[2]:
            n = 0
            while pixel_array[self.pos[0] - 17 - self.vel + n][self.pos[1]] == -1 \
                    or pixel_array[self.pos[0] - 17 - self.vel + n][self.pos[1] + 17] == -1 \
                    or pixel_array[self.pos[0] - 17 - self.vel + n][self.pos[1] - 17] == -1:
                n += 1
            self.pos[0] -= self.vel - n
        if movement[0]:
            n = 0
            while pixel_array[self.pos[0] + 17 + self.vel - n][self.pos[1]] == -1 \
                    or pixel_array[self.pos[0] + 17 + self.vel - n][self.pos[1] + 17] == -1 \
                    or pixel_array[self.pos[0] + 17 + self.vel - n][self.pos[1] - 17] == -1:
                n += 1
            self.pos[0] += self.vel - n
        if movement[3]:
            n = 0
            while pixel_array[self.pos[0]][self.pos[1] - 17 - self.vel + n] == -1 \
                    or pixel_array[self.pos[0] + 17][self.pos[1] - 17 - self.vel + n] == -1 \
                    or pixel_array[self.pos[0] - 17][self.pos[1] - 17 - self.vel + n] == -1:
                n += 1
            self.pos[1] -= self.vel - n

        if movement[1]:
            n = 0
            while pixel_array[self.pos[0]][self.pos[1] + 17 + self.vel - n] == -1 \
                    or pixel_array[self.pos[0] + 17][self.pos[1] + 17 + self.vel - n] == -1 \
                    or pixel_array[self.pos[0] - 17][self.pos[1] + 17 + self.vel - n] == -1:
                n += 1
            self.pos[1] += self.vel - n

    def draw(self):
        individuals_surf.set_alpha(255)
        pygame.draw.rect(individuals_surf, (0, 0, 0), ((0, 0), (35, 35)))
        pygame.draw.rect(individuals_surf, self.color, ((5, 5), (25, 25)))

    def _draw_ray_casting(self, coordinates, colors):
        for n, coordinate in enumerate(coordinates):
            pygame.draw.line(game_surf, colors[n], (self.pos[0], self.pos[1]),
                             (coordinate[0], coordinate[1]), width=1)

    @staticmethod
    def copy(_individual):
        __individual = Individual(deepcopy(initial_position), _individual.vel, _individual.vision_rays,
                                  deepcopy(
                                      _individual.neural_network), _individual.draw_ray_casting,
                                  _individual.output_function, deepcopy(_individual.color))
        __individual.fitness = deepcopy(_individual.fitness)
        return __individual


class Obstacle:
    def __init__(self, pos, vel, vel_function):
        self.pos = pos
        self.vel = vel
        self.next_pos = pos
        self.vel_function = vel_function
        self.draw_collision()
        self.draw_color()

    def draw_collision(self):
        self.pos = self.next_pos
        self.vel_function(self)
        self.next_pos = self.pos + self.vel
        pygame.draw.circle(obstacles_surf, np.array([0, 0, 0]), self.pos, 13)

    def draw_color(self):
        pygame.draw.circle(game_surf, (0, 0, 255), self.pos, 7)


def get_fit_array(final_pos):
    map_array = pygame.surfarray.pixels3d(map_surf)
    size = map_surf.get_size()
    _fit_array = np.zeros([size[0], size[1]])
    evaluating_pixels = [final_pos]
    fit_value = 1
    while evaluating_pixels:
        new_evaluating_pixels = []
        for pixel in evaluating_pixels:
            testing_pixels = np.array([np.array([pixel[0], pixel[1] - 1]),
                                       np.array([pixel[0], pixel[1] + 1]),
                                       np.array([pixel[0] - 1, pixel[1]]),
                                       np.array([pixel[0] + 1, pixel[1]]),
                                       np.array([pixel[0] - 1, pixel[1] - 1]),
                                       np.array([pixel[0] + 1, pixel[1] - 1]),
                                       np.array([pixel[0] - 1, pixel[1] + 1]),
                                       np.array([pixel[0] + 1, pixel[1] + 1])])
            for testing_pixel in testing_pixels:
                if _fit_array[testing_pixel[0]][testing_pixel[1]] == 0 \
                        and np.array([testing_pixel[0], testing_pixel[1]] != final_pos).any():
                    if (map_array[testing_pixel[0]][testing_pixel[1]] == np.array([0, 0, 0])).all():
                        _fit_array[testing_pixel[0]][testing_pixel[1]] = -1
                    else:
                        _fit_array[testing_pixel[0]
                                   ][testing_pixel[1]] = fit_value
                        new_evaluating_pixels.append(testing_pixel)
        evaluating_pixels = new_evaluating_pixels
        fit_value += 1
    _fit_array[_fit_array == 0] = -1
    _fit_array[final_pos[0]][final_pos[1]] = 0
    return _fit_array


def get_next_gen(_best_parent):
    if len(historical_fitnesses) > 0:
        best_fitness = best_parent.fitness
    else:
        best_fitness = -1
    for n, child in enumerate(childs[elite:]):
        if child.alive:
            child.get_fitness()
        child.alive = True
        if child.fitness < parents[n].fitness:
            if parents[n].age > max_age or len(historical_fitnesses) == 0:
                parents[n] = random_individual()
                continue
            childs[n] = Individual.copy(parents[n])
            if len(historical_fitnesses) > 0:
                index = bisect_left(
                    historical_fitnesses, individual.fitness, 0, len(historical_fitnesses))
                difference = len(historical_fitnesses) - index
                proportionSimilar = difference / len(historical_fitnesses)
                if random.random() < np.e ** (-proportionSimilar):
                    parents[n] = deepcopy(child)
                    parents[n].age += 1
                    continue
                parents[n] = deepcopy(_best_parent)
                parents[n].age = 0
        else:
            parents[n] = deepcopy(child)
            parents[n].age = 0
        if len(historical_fitnesses) == 0 or child.fitness > best_fitness:
            historical_fitnesses.append(child.fitness)
            _best_parent = deepcopy(child)
            best_fitness = deepcopy(_best_parent.fitness)
            # _best_parent.neural_network.write_data(document=f'{_best_parent.neural_network.name}_{len(historical_fitnesses)}')
    for n, child in enumerate(childs[:elite]):
        if child.fitness > best_fitness:
            historical_fitnesses.append(child.fitness)
            _best_parent = deepcopy(child)
            best_fitness = deepcopy(_best_parent.fitness)
    for n, child in enumerate(childs[:elite]):
        parents[n] = deepcopy(_best_parent)
    for n, parent in enumerate(parents):
        strategy = random.choice(strategies)
        childs[n] = strategy(parent)
    for child in childs:
        child.fitness = -1
        child.pos = deepcopy(initial_position)
        child.higher_fitness = -1
        child.higher_fitness_time = None
    return historical_fitnesses, _best_parent


def mutate_strategy(parent):
    child = deepcopy(parent)
    rand_range = random.randint(1, 10)
    for _ in range(rand_range):
        child.neural_network.mutate()
    child.color[random.randint(0, 2)] = random.randint(0, 255)
    return child


def uniform_mutate_strategy(parent):
    child = deepcopy(parent)
    rand_range = random.randint(1, 10)
    for _ in range(rand_range):
        child.neural_network.uniform_mutate()
    child.color[random.randint(0, 2)] = random.randint(0, 255)
    return child


def layer_crossover_strategy(parent):
    donor = random.choice(parents)
    while donor == parent:
        donor = random.choice(parents)
    child = deepcopy(parent)
    child.neural_network = layer_crossover(
        parent.neural_network, donor.neural_network, True)
    donor_color_index = random.randint(0, 2)
    child.color[donor_color_index] = donor.color[donor_color_index]
    return child


def n_crossover_strategy(parent):
    donor = random.choice(parents)
    while donor == parent:
        donor = random.choice(parents)
    child = deepcopy(donor)
    child.neural_network = n_crossover(
        parent.neural_network, donor.neural_network, True)
    donor_color_index = random.randint(0, 2)
    child.color[donor_color_index] = donor.color[donor_color_index]
    return child


# @cuda.jit(target='cuda', forceobj=True)
# def _run(individual, fit_array, obstacles_collision_array, game_surf_collision_array):
#    individual.run(fit_array, obstacles_collision_array, game_surf_collision_array)


def random_individual():
    return Individual(deepcopy(initial_position), individual_vel, rays,
                      random_homogeneous_network([rays * 2 + 8, 11, 11, 4], neurons_function='relu',
                                                 neurons_second_step=None, weights=True, bias=False,
                                                 rand_weight_range=10, rand_bias_range=None,
                                                 custom_function=False, custom_second_step=False,
                                                 name=True),
                      output_function=Neuron.relu, draw_ray_casting=True)


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    background_color = (171, 164, 255)
    screen.fill(background_color)
    pygame.display.set_caption('game')
    game_surf = pygame.surface.Surface((922, 522))
    map_surf = pygame.surface.Surface((922, 522))
    individuals_surf = pygame.surface.Surface((35, 35))
    obstacles_surf = pygame.surface.Surface((922, 522))
    obstacles_surf.fill([255, 255, 255])
    map_surf.fill([255, 255, 255])
    obstacles_surf.set_colorkey([255, 255, 255])
    individuals_surf.fill([255, 255, 255])
    individuals_surf.set_colorkey([255, 255, 255])
    checkpoint_color = (168, 235, 169)
    clock = pygame.time.Clock()
    run, initial_position = level_1()
    run.draw_background()
    run.draw_walls()
    game_surf.blit(map_surf, (0, 0))
    game_surf_collision_array = pygame.surfarray.pixels_red(game_surf)
    fit_array = np.loadtxt('level_1_fit_array.txt', dtype=int)
    # fit_array = get_fit_array(np.array([836, 286]))
    # np.savetxt('level_1_fit_array.txt', fit_array, fmt='%d')
    # pygame.display.update()
    # test = get_fit_array(np.array([836, 286]))
    # df = pd.DataFrame(test)
    # df.to_excel('test.xlsx', index=False)
    pool_size = 40
    elite = 5
    individual_vel = 4  # MIN 1, MAX 11
    rays = 16
    #clifford = load_data(document='Clifford Calvin Ponce Adams Marciano Cooper Granado Harris Macinnes Turner Stewart Varghese Castro Markle Novotny Dow Escamilla Patock Beason Cogburn_102', name='Clifford Calvin Ponce Adams Marciano Cooper Granado Harris Macinnes Turner Stewart Varghese Castro Markle Novotny Dow Escamilla Patock Beason Cogburn')
    #parents = [random_individual() for _ in range(pool_size)]
    parents = [random_individual() for _ in range(pool_size)]
    del game_surf_collision_array
    game_surf.blit(obstacles_surf, (0, 0))
    run.obstacles_color()
    screen.blit(game_surf, (500, 280))
    childs = deepcopy(parents)
    best_parent = None
    historical_fitnesses = []
    strategies = [lambda parent: n_crossover_strategy(parent),
                  lambda parent: layer_crossover_strategy(parent),
                  lambda parent: mutate_strategy(parent),
                  lambda parent: uniform_mutate_strategy(parent)]
    max_time = 10
    max_age = 10
    generation = 0
    pygame.display.update()
    while True:
        run, initial_position = level_1()
        generation += 1
        run.draw_background()
        run.draw_walls()
        game_surf.blit(map_surf, (0, 0))
        screen.blit(game_surf, (500, 280))
        start_time = time.time()
        while time.time() <= max_time + start_time:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_DOWN:
                        max_time -= 1
                        print('max_time:', max_time)
                    if event.key == K_UP:
                        max_time += 1
                        print('max_time:', max_time)
            run.draw_background()
            run.draw_walls()
            game_surf.blit(map_surf, (0, 0))
            run.obstacles()
            game_surf.blit(obstacles_surf, (0, 0))
            obstacles_collision_array = pygame.surfarray.pixels_red(
                obstacles_surf)
            game_surf_collision_array = pygame.surfarray.pixels_red(game_surf)
            for individual in childs:
                if individual.alive:
                    #_run(individual, fit_array, obstacles_collision_array, game_surf_collision_array)
                    individual.run(
                        fit_array, obstacles_collision_array, game_surf_collision_array)
            del obstacles_collision_array
            del game_surf_collision_array
            run.obstacles_color()
            screen.blit(game_surf, (500, 280))
            for individual in childs:
                individual.blit()
            pygame.display.update()
        obstacles_collision_array = pygame.surfarray.pixels_red(obstacles_surf)
        game_surf_collision_array = pygame.surfarray.pixels_red(game_surf)
        historical_fitnesses, best_parent = get_next_gen(best_parent)
        del obstacles_collision_array
        del game_surf_collision_array
