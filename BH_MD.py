import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800,600
screen = pygame.display.set_mode((WIDTH, HEIGHT),pygame.RESIZABLE)
fps = 60
pygame.display.set_caption(f'Ideal Gas Simulation with Barnes-Hut FPS: {fps}')

# Particle properties
N = 1000  # Number of particles
PARTICLE_RADIUS = 2
VELOCITY_RANGE = 20  # Maximum initial velocity in any direction

# Barnes-Hut parameters
THETA = 3.0  # Threshold parameter for Barnes-Hut approximation

# Define parameters of Lennard-Jones potential
LJ_sigma = 5.0
LJ_epsilon = 10000.0
PARTICLE_MASS = 1.0

# Particle class
class Particle:
    def __init__(self, id, mass):
        self.x = random.uniform(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        self.y = random.uniform(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)
        self.vx = random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE)
        self.vy = random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE)
        self.fx = 0
        self.fy = 0
        self.id = id
        self.mass = mass
        self.radius = PARTICLE_RADIUS

    def move(self, dt):
        self.vx += self.fx / self.mass * dt
        self.vy += self.fy / self.mass * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Apply periodic boundary conditions
        if self.x < 0:
            self.x = 0
            self.vx *= -1.0000001
        elif self.x > WIDTH:
            self.x = WIDTH
            self.vx *= -1.0000001
        if self.y < 0:
            self.y = 0
            self.vy *= -1.0000001
        elif self.y > HEIGHT:
            self.y = HEIGHT
            self.vy *= -1.0000001

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius)

# QuadTree class for Barnes-Hut algorithm
class QuadTree:
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.particles = []
        self.divided = False

    def insert(self, particle):
        if not (self.x0 <= particle.x < self.x0 + self.width and self.y0 <= particle.y < self.y0 + self.height):
            return False

        if len(self.particles) < 1 and not self.divided:
            self.particles.append(particle)
            return True

        if not self.divided:
            self.subdivide()

        return (self.nw.insert(particle) or self.ne.insert(particle) or
                self.sw.insert(particle) or self.se.insert(particle))

    def subdivide(self):
        self.divided = True
        w, h = self.width / 2, self.height / 2
        self.nw = QuadTree(self.x0, self.y0, w, h)
        self.ne = QuadTree(self.x0 + w, self.y0, w, h)
        self.sw = QuadTree(self.x0, self.y0 + h, w, h)
        self.se = QuadTree(self.x0 + w, self.y0 + h, w, h)

    def calculate_force(self, particle, theta):
        if len(self.particles) == 1 and self.particles[0] is particle:
            return 0, 0

        dx = (self.x0 + self.width / 2) - particle.x
        dy = (self.y0 + self.height / 2) - particle.y
        distance = np.sqrt(dx * dx + dy * dy)
        if self.divided:
            s = self.width
            if s / distance < theta:
                fx, fy = self.approximate_force(particle)
                return fx, fy
            else:
                fx_nw, fy_nw = self.nw.calculate_force(particle, theta)
                fx_ne, fy_ne = self.ne.calculate_force(particle, theta)
                fx_sw, fy_sw = self.sw.calculate_force(particle, theta)
                fx_se, fy_se = self.se.calculate_force(particle, theta)
                return fx_nw + fx_ne + fx_sw + fx_se, fy_nw + fy_ne + fy_sw + fy_se
        else:
            fx, fy = 0, 0
            for p in self.particles:
                if p is not particle:
                    dx = p.x - particle.x
                    dy = p.y - particle.y
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance > 2 * PARTICLE_RADIUS:
                        force_magnitude = -LJ_epsilon * ((LJ_sigma / distance)**2)
                        fx += force_magnitude * dx / distance
                        fy += force_magnitude * dy / distance
            return fx, fy

    def approximate_force(self, particle):
        dx = (self.x0 + self.width / 2) - particle.x
        dy = (self.y0 + self.height / 2) - particle.y
        distance = np.sqrt(dx * dx + dy * dy)
        force_magnitude = -LJ_epsilon * ((LJ_sigma / distance)**2)
        fx = force_magnitude * dx / distance
        fy = force_magnitude * dy / distance
        return fx, fy

# Create particles
particles = [Particle(i, PARTICLE_MASS) for i in range(N)]

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    dt = clock.get_time()/1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h

    # Create QuadTree
    quad_tree = QuadTree(0, 0, WIDTH, HEIGHT)
    for particle in particles:
        quad_tree.insert(particle)

    # Calculate forces and move particles
    for particle in particles:
        fx, fy = quad_tree.calculate_force(particle, THETA)
        particle.fx = fx
        particle.fy = fy
        particle.move(dt)

    # Draw particles
    screen.fill((0, 0, 0))  # Clear screen with black color
    for particle in particles:
        particle.draw(screen)
    fps = clock.get_fps()
    pygame.display.set_caption(f'Ideal Gas Simulation with Barnes-Hut FPS: {fps:.2f}')

    pygame.display.flip()
    clock.tick(60)  # Cap the frame rate at 60 FPS

pygame.quit()

