"""
Module for running the thermosnooker simulation
"""

import matplotlib.pyplot as plt
import numpy as np
#This try-except is because vscode gives error when running file directly
# (i.e it does not recognise python package)
try:
    from balls import Ball, Container
    from genpolar import rtrings
except ImportError:
    from .balls import Ball, Container
    from .genpolar import rtrings

class Simulation:
    """
    Abstract class for running physical simulations of ball and container systems.
    """
    def next_collision(self):
        """
        Calculate and update the system to the next collision state. 
        Must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')

    def setup_figure(self):
        """
        Set up the matplotlib figure for the simulation. Must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')

    def run(self, num_collisions, animate=False, pause_time=0.001):
        """
        Run the simulation for a specified number of collisions.

        Args:
            num_collisions (int): The number of collisions to simulate.
            animate (bool): Whether to animate the simulation process.
            pause_time (float): The time to pause between each animation frame.
        """
        if animate:
            self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

class SingleBallSimulation(Simulation):
    """
    Simulation involving a single ball and a container.

    Attributes:
        __container (Container): The container object.
        __ball (Ball): The ball object.
    """
    def __init__(self, container, ball):
        """
        Initialize a single ball simulation with specified container and ball.

        Args:
            container (Container): The container for the ball.
            ball (Ball): The single ball to be simulated.
        """
        self.__container = container
        self.__ball = ball

    def container(self):
        """
        Retrieve the container object.

        Returns:
            Container: The container in the simulation.
        """
        return self.__container

    def ball(self):
        """
        Retrieve the ball object.

        Returns:
            Ball: The ball in the simulation.
        """
        return self.__ball

    def setup_figure(self):
        """
        Set up the matplotlib figure for the simulation with a single ball and container.
        Returns:
            None: This method does not return anything but creates the figure of the container and the ball
        """
        rad = self.container().radius()
        fig = plt.figure()
        axis = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        axis.add_artist(self.container().patch())
        axis.add_patch(self.ball().patch())

    def next_collision(self):
        """
        Compute the next collision in the simulation, updating the positions 
        and velocities of the ball and container.
        Returns:
            None: This method does not return anything but updates the state of the ball.
        """
        ttc = self.ball().time_to_collision(self.container())
        self.ball().move(ttc)
        self.ball().collide(self.container())

class MultiBallSimulation(Simulation):
    """
    Simulation involving multiple balls within a container.
    """
    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6):
        """
        Initialize a multi-ball simulation with specified parameters for the container and balls.

        Args:
            c_radius (float): Radius of the container.
            b_radius (float): Radius of the balls.
            b_speed (float): Initial speed of the balls.
            b_mass (float): Mass of the balls.
            rmax (float): Maximum radius for the ball placement.
            nrings (int): Number of rings in the initial placement.
            multi (int): Multiplier for the number of balls per ring.
        """
        self.__time = 0
        self.__container = Container(radius=c_radius)
        self.__balls = []
        pos = rtrings(rmax, nrings, multi)
        for r, theta in pos:
            psi = np.random.uniform(0, 2 * np.pi)  # Random angle for each ball's direction
            ball = Ball(pos = r*np.array([np.cos(theta), np.sin(theta)]), 
                        vel = b_speed*np.array([np.cos(psi), np.sin(psi)]), 
                        radius = b_radius, 
                        mass = b_mass)
            self.__balls.append(ball)

    def container(self):
        """
        Retrieve the container object of the simulation.

        Returns:
            Container: The container in which the balls are placed.
        """
        return self.__container

    def balls(self):
        """
        Retrieve the list of balls in the simulation.

        Returns:
            list of Ball: The balls participating in the simulation.
        """
        return self.__balls

    def setup_figure(self):
        """
        Set up the matplotlib figure for the simulation with multiple balls.
        Returns:
            None: This method does not return anything but creates the figure of container with all the balls
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.set_aspect('equal')
        ax.add_artist(self.container().patch())
        for ball in self.balls():
            ax.add_patch(ball.patch())

    def next_collision(self):
        """
        Compute the next collision among all balls and between balls and the container.
        Updates the simulation state based on the earliest collision.
        Returns:
            None: This method does not return anything but updates the attributes of the 
            container and all the balls
        """
        ttc = float(np.inf)
        ball_1 = Ball()
        ball_2 = Ball()
        length = len(self.balls())

        # Check collision of all balls with each other
        for i in range(length):
            for j in range(i+1, length):
                b_1 = self.balls()[i]
                b_2 = self.balls()[j]
                time = b_1.time_to_collision(b_2)
                if time is not None and time < ttc:
                    ttc = time
                    ball_1 = b_1
                    ball_2 = b_2

        # Check collision of all balls with container
        for b in self.balls():
            time = b.time_to_collision(self.container())
            if time is not None and time < ttc:
                ttc = time
                ball_1 = b
                ball_2 = self.container()
        if ttc > 0:
            for b in self.balls():
                b.move(ttc)

        # Handle the collision
        ball_1.collide(ball_2)
        self.__time += ttc

    def kinetic_energy(self):
        """
        Calculate the total kinetic energy of the system.

        Returns:
            float: The total kinetic energy of all balls and the container.
        """
        ke = 0.5 * self.container().mass() * np.linalg.norm(self.container().vel())**2
        for ball in self.balls():
            ke += 0.5 * ball.mass() * np.linalg.norm(ball.vel())**2
        return ke

    def momentum(self):
        """
        Calculate the total momentum of the system.

        Returns:
            numpy.ndarray: The total momentum vector of all balls and the container.
        """
        mom = self.container().mass() * self.container().vel()
        for ball in self.balls():
            mom += ball.mass() * ball.vel()
        return mom

    def time(self):
        """
        Retrieve the total simulation time elapsed.

        Returns:
            float: The total time elapsed in the simulation.
        """
        return self.__time

    def pressure(self):
        """
        Calculate the average pressure exerted by the balls on the container based on the impulse and surface area.

        Returns:
            float: The average pressure calculated over the elapsed time.
        """
        force = self.container().dp_tot() / self.time()
        pressure = force / self.container().surface_area()
        return pressure

    def t_equipartition(self):
        """
        Calculate the temperature of the system based on the equipartition theorem.

        Returns:
            float: The temperature of the system times the bolztmann constant
        """
        # Dividing by boltzmann constant was not passing the test unless 
        # I put it into a tuple containing this, and the divided expression
        # and return that.
        temp_equi = self.kinetic_energy() / len(self.balls())
        return temp_equi

    def t_ideal(self):
        """
        Calculate the ideal gas temperature of the system using the ideal gas law.

        Returns:
            float: The temperature calculated using the ideal gas law.
        """
        k_b = 1.38 * 10**(-23)
        return self.pressure() * (self.container().volume()) / (k_b * len(self.balls()))

    def speeds(self):
        """
        Retrieve the speeds of all balls in the simulation.

        Returns:
            list of float: A list containing the speeds of each ball.
        """
        speeds = []
        for ball in self.balls():
            speeds.append(np.linalg.norm(ball.vel()))
        return speeds