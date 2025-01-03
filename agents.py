import math
import random
import pygame
from constants import *

# Constants (assuming some color constants like GREEN, YELLOW, etc. are defined somewhere)
class Agent:
    def __init__(self, x, y, colour, radius, speed, angle, vision_distance, vision_angle, stamina_max, agility):
        self.x = x
        self.y = y
        self.colour = colour
        self.base_speed = speed  # Maximum base speed
        self.current_speed = 0  # Start with zero speed
        self.acceleration = 0.2  # Rate of acceleration
        self.deceleration = 0.1  # Rate of deceleration
        self.angle = angle  # Initial angle (direction the agent is facing)
        self.radius = radius
        self.vision_distance = vision_distance
        self.vision_angle = vision_angle
        self.stamina_max = stamina_max # since it will be an evolving factor
        self.stamina = stamina_max  # Start with full stamina
        self.agility = agility  # Agility factor (affects turning speed)
        self.dx = math.cos(math.radians(self.angle)) * self.current_speed
        self.dy = math.sin(math.radians(self.angle)) * self.current_speed
        self.resting = False
        self.wander_timer = 0 # to create more natural wandering movement
        self.angular_velocity = 0
        self.turn_smoothness = 0.1 # we want smooth turns

    def move(self, screen, is_active=False, is_chased=False):
        # Speed adjustment based on stamina
        stamina_factor = self.stamina / self.stamina_max  # Normalize stamina (0 to 1)
        max_speed = self.base_speed * (0.5 + stamina_factor * 0.5)  # Minimum speed is 50% of base speed
        min_speed = 0.4 * self.base_speed  # Define a minimum speed as 20% of the base speed

        # Add random speed fluctuation
        speed_fluctuation = random.uniform(-0.05, 0.05) * max_speed  # Fluctuate up to ±5% of max speed

        if self.stamina <= 10 and not is_chased:  # If stamina is 0 and not being chased, start resting
            self.resting = True
            self.current_speed = 0  # Stop moving while resting
            self.stamina += 0.5  # Regain stamina faster while resting
            if self.stamina >= self.stamina_max * 0.5:  # Resume moving when stamina is at least 50%
                self.resting = False
        elif is_active and self.stamina > 10:  # Only accelerate if active and stamina allows
            if self.current_speed < max_speed:
                self.current_speed += self.acceleration  # Accelerate until reaching max speed
            self.resting = False  # Ensure agent is not resting when active
        else:  # Decelerate when inactive or out of stamina
            if self.current_speed > min_speed:
                self.current_speed -= self.deceleration
            elif self.current_speed < min_speed:  # Ensure the speed doesn't drop below min_speed
                self.current_speed = min_speed
                self.wander()  # Start wandering behavior when speed is at its minimum

        # Apply fluctuation to current speed
        self.current_speed = max(min_speed, self.current_speed + speed_fluctuation)  # Ensure speed doesn't go below min_speed

        # Update position using the current speed
        self.dx = math.cos(math.radians(self.angle)) * self.current_speed
        self.dy = math.sin(math.radians(self.angle)) * self.current_speed
        self.x += self.dx
        self.y += self.dy

        # Stamina consumption logic
        if is_active:
            self.stamina -= 2 * (self.current_speed / self.base_speed)
        else:
            if self.resting:
                self.stamina += 0.3  # Regain stamina more slowly while resting
            else:
                self.stamina += 0.05  # Slow recovery when not actively chasing
        

        # Ensure stamina stays within bounds
        self.stamina = max(0, min(self.stamina, self.stamina_max))

        self._handle_border_collision(screen)
        self.angle += self.angular_velocity
        self.update_direction()


    def update_direction(self):
        self.dx = math.cos(math.radians(self.angle)) * self.current_speed
        self.dy = math.sin(math.radians(self.angle)) * self.current_speed



    def turn(self, target_angle):
        # Calculate the difference in angle (shortest path)
        angle_diff = (target_angle - self.angle + 180) % 360 - 180

        # If the angle difference is small enough, stop turning
        if abs(angle_diff) < 1:
            self.angular_velocity = 0
            return

        # Adjust angular velocity based on agility and current speed
        turn_speed = self.agility * (1.0 / (1 + self.current_speed * 0.1))  # Higher speed reduces turning speed
        self.angular_velocity = angle_diff * turn_speed * self.turn_smoothness  # Smooth the turn

        # Limit angular velocity to prevent overshooting
        max_turn_speed = 10  # Maximum turning speed
        if self.angular_velocity > max_turn_speed:
            self.angular_velocity = max_turn_speed
        elif self.angular_velocity < -max_turn_speed:
            self.angular_velocity = -max_turn_speed


    def _handle_border_collision(self, screen):
        # Adjust position if agent hits the screen border
        if self.x - self.radius < 0:
            self.x = self.radius
            self.dx = abs(self.dx)
            self.angle = 0
            self.update_direction()
        elif self.x + self.radius > screen.get_width():
            self.x = screen.get_width() - self.radius
            self.dx = -abs(self.dx)
            self.angle = 180
            self.update_direction()

        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy = abs(self.dy)
            self.angle = 90
            self.update_direction()
        elif self.y + self.radius > screen.get_height():
            self.y = screen.get_height() - self.radius
            self.dy = -abs(self.dy)
            self.angle = 270
            self.update_direction()

    def wander(self):
        self.angular_velocity = 0
        self.wander_timer += 1
        if self.wander_timer > random.randint(50, 500):
            random_angle = random.uniform(-15, 15)
            self.turn(self.angle + random_angle)
            self.wander_timer = 0

        # Add stamina consumption while wandering
        self.stamina -= 0.1  # Decrease stamina slowly while wandering


    def draw(self, screen):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.radius)

        # Draw vision cone
        start_angle = math.radians(self.angle - self.vision_angle / 2)
        end_angle = math.radians(self.angle + self.vision_angle / 2)

        vision_points = [
            (self.x, self.y),
            (self.x + math.cos(start_angle) * self.vision_distance,
             self.y + math.sin(start_angle) * self.vision_distance),
            (self.x + math.cos(end_angle) * self.vision_distance,
             self.y + math.sin(end_angle) * self.vision_distance),
        ]
        pygame.draw.polygon(screen, self.colour, vision_points, 1)

        # Draw stamina bar
        bar_color = GREEN if self.stamina > 30 else YELLOW if self.stamina > 10 else RED
        pygame.draw.rect(screen, bar_color, (self.x - 10, self.y - 20, self.stamina / 2, 5))


class Predator(Agent):
    def chase(self, prey_list, screen):
        closest_prey = None
        closest_distance = float('inf')

        # Find the closest prey in vision
        for prey in prey_list:
            distance = math.sqrt((self.x - prey.x)**2 + (self.y - prey.y)**2)  # Evaluate distance to each prey
            angle_to_prey = math.degrees(math.atan2(prey.y - self.y, prey.x - self.x)) % 360  # Angle difference to prey

            if distance < closest_distance and self.in_vision(angle_to_prey):
                closest_prey = prey
                closest_distance = distance

        # If prey is found within vision, move towards it
        if closest_prey:
            self.angle = math.degrees(math.atan2(closest_prey.y - self.y, closest_prey.x - self.x))  # Face the prey
            self.update_direction()
            self.move(screen, is_active=True)  # Ensure stamina consumption during active chase
        else:
            self.wander()

    def in_vision(self, angle_to_target):
        delta_angle = (angle_to_target - self.angle + 360) % 360  # Difference in angle
        return delta_angle <= self.vision_angle / 2 or delta_angle >= 360 - self.vision_angle / 2


class Prey(Agent):
    def __init__(self, x, y, colour, radius, speed, angle, vision_distance, vision_angle, stamina_max, agility):
        super().__init__(x, y, colour, radius, speed, angle, vision_distance, vision_angle, stamina_max, agility)
        self.evasion_timer = 0
        self.evasion_offset = 0  # Persistent random offset during evasion

    def evade(self, predator_list, screen):
        closest_predator = None
        closest_distance = float('inf')

        # Find the closest predator within vision range
        for predator in predator_list:
            distance = math.sqrt((self.x - predator.x)**2 + (self.y - predator.y)**2)  # Fixed distance calculation
            if distance < self.vision_distance and distance < closest_distance:
                closest_predator = predator
                closest_distance = distance

        # If a predator is detected, move away
        if closest_predator:
            self.resting = False
            # evasion timer so that directions can be varied while fleeing to simulate more realistic movement
            if self.evasion_timer <= 0:
                self.evasion_offset = random.uniform(-30, 30)
                self.evasion_timer = random.randint(20, 60)

            angle_to_predator = (math.degrees(math.atan2(self.y - closest_predator.y, self.x - closest_predator.x)))
            target_angle = (angle_to_predator + self.evasion_offset) % 360  # Move opposite to predator
            self.turn(target_angle)
            self.update_direction()

            self.evasion_timer -= 1
            self.move(screen, is_active=True, is_chased=True)
        else:
            # Default to wandering if no predators are near
            self.wander()

