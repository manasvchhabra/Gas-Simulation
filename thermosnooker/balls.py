"""
Module for the thermodynamic balls
"""
import numpy as np
from matplotlib.patches import Circle
class Ball():
    """
    Represents the ball object
    """
    def __init__(self, pos=None, vel=None, radius=1., mass=1.):
        """
        Initializes a new Ball instance with position, velocity, radius, and mass.

        Args:
            pos (list or numpy.ndarray): The initial position of the ball, defaults to [0., 0.].
            vel (list or numpy.ndarray): The initial velocity of the ball, defaults to [1., 0.].
            radius (float): The radius of the ball, defaults to 1.
            mass (float): The mass of the ball, defaults to 1.

        Raises:
            ValueError: If pos or vel are not list or numpy.ndarray or their lengths are not 2.
        """
        if pos is None:
            pos = [0., 0.]
        if vel is None:
            vel = [1., 0.]
        if not isinstance(pos, (list, np.ndarray)):
            raise ValueError("Position must be a list or numpy array")
        if len(pos) != 2:
            raise ValueError("Position must be a list or numpy array of length 2")

        if not isinstance(vel, (list, np.ndarray)):
            raise ValueError("Velocity must be a list or numpy array")
        if len(vel) != 2:
            raise ValueError("Velocity must be a list or numpy array of length 2")

        self.__pos = np.array(pos, dtype=float)
        self.__vel = np.array(vel, dtype=float)
        self.__radius = radius
        self.__mass = mass
        self.__patch = Circle(xy=(self.pos()[0], self.pos()[1]), radius=self.__radius, color='r')

    def pos(self):
        """
        Retrieves the current position of the ball.

        Returns:
            numpy.ndarray: The current position of the ball.
        """
        return self.__pos
    def vel(self):
        """
        Retrieves the current velocity of the ball.

        Returns:
            numpy.ndarray: The current velocity of the ball.
        """
        return self.__vel
    def radius(self):
        """
        Retrieves the radius of the ball.

        Returns:
            float: The radius of the ball.
        """
        return self.__radius  
    def mass(self):
        """
        Retrieves the mass of the ball.

        Returns:
            float: The mass of the ball.
        """
        return self.__mass

    def set_vel(self, vel):
        """
        Sets a new velocity for the ball.

        Args:
            vel (list or numpy.ndarray): The new velocity for the ball.

        Raises:
            ValueError: If vel is not a list or numpy.ndarray or its length is not 2.
        """
        if not isinstance(vel, (list, np.ndarray)):
            raise ValueError("Velocity must be a list or numpy array")
        if len(vel) != 2:
            raise ValueError("Velocity must be a list or numpy array of length 2")
        self.__vel = np.array(vel)

    def move(self, dt):
        """
        Moves the ball by updating its position based on its velocity and the time delta.

        Args:
            dt (float): The time interval over which the ball moves.
        """
        if dt is not None:
            self.__pos = self.__pos + self.__vel*dt
            self.__patch.set_center((float(self.pos()[0]), float(self.pos()[1])))

    def patch(self):
        """
        Retrieves the matplotlib patch of the ball for visualization.

        Returns:
            matplotlib.patches.Circle: The visual representation of the ball.
        """
        return self.__patch  
    def time_to_collision(self, other):
        """
        Calculates the time until this ball will collide with another ball.

        Args:
            other (Ball): The other ball to check for potential collision.

        Returns:
            float: The time until collision, or None if no collision is predicted.
        """
        #The relative position and velocities of self and other
        r, v = self.pos() - other.pos(), self.vel() - other.vel()

        if isinstance(self, Container) or isinstance(other, Container):
            R = self.radius() - other.radius()
        else:
            R = self.radius() + other.radius()


        a = np.dot(v,v)
        b = 2 * np.dot(r, v)
        c = np.dot(r, r) - R**2
        discr = b**2 - 4*a*c #discriminant
        if discr < 0:
            return None #No real solutions
        if a == 0:
            if b == 0:
                return None 
            dt = -c / b
            return dt if dt > 0 else None

        dt1 = (-b - np.sqrt(discr)) / (2*a)
        dt2 = (-b + np.sqrt(discr)) / (2*a)
        # Return the smallest positive time
        tolerance = 1e-6 #Tolerance to avoid floating point errors
        if dt1 > tolerance and dt2 > tolerance:
            return dt1 #i.e the smaller of two
        if dt1 > tolerance:
            return dt1
        if dt2 > tolerance:
            return dt2
        return None
    def collide(self, other):
        """
            Processes the collision between this ball and another object.

            Args:
                other (Ball or Container): The object this ball is colliding with. This can be either 
                                        another Ball or a Container.

            Return:
                None: This method does not return anything but updates the attributes of the objects.
        """
        #Initial parameters of objects stored before making changes
        u_1 = self.vel()
        u_2 = other.vel()
        m_1 = self.mass()
        m_2 = other.mass()
        x_1 = self.pos()
        x_2 = other.pos()
        if isinstance(other, Container):
            v_1 = self.vel()
            v_2 = other.vel()
            normal = self.pos() / np.linalg.norm(self.pos())
            self.set_vel(self.vel() - 2 * np.dot(self.vel(), normal) * normal)

            dp = (self.vel() - v_1)*self.mass()
            other._impulse += float(np.linalg.norm(dp))
            other.set_vel(v_2-dp/other.mass())
        else:
            v_1 = u_1 - 2*m_2/(m_1+m_2)*np.dot(u_1-u_2, x_1-x_2)/np.linalg.norm(x_1-x_2)**2 *(x_1-x_2)
            v_2 = u_2 - 2*m_1/(m_1+m_2)*np.dot(u_2-u_1, x_2-x_1)/np.linalg.norm(x_2-x_1)**2 *(x_2-x_1)
            self.set_vel(v_1)
            other.set_vel(v_2)

class Container(Ball):
    """ 
    Represents a large container that contains the balls.
    """
    def __init__(self, radius=10., mass=10000000.):
        """
        Initializes a new Container instance, inheriting from Ball.

        Args:
            radius (float): The radius of the container, defaults to 10.
            mass (float): The hypothetical mass of the container, defaults to 10000000.
        """
        Ball.__init__(self, pos=[0,0], vel=[0,0], radius=radius, mass=mass)
        self._impulse = 0.
        #Patch for container, separate from patch from ball class.
        self.__newpatch = Circle(xy=(self.pos()[0], self.pos()[1]), radius=self.radius(), color='b', fill=False, linewidth=5)
    def volume(self):
        """
        Calculates the volume of the container.

        Returns:
            float: The volume of the container.
        """
        return np.pi * self.radius()**2

    def surface_area(self):
        """
        Calculates the surface area of the container.

        Returns:
            float: The surface area of the container.
        """
        return 2 * np.pi * self.radius()

    def patch(self):
        """
        Retrieves the matplotlib patch of the container for visualization.

        Returns:
            matplotlib.patches.Circle: The visual representation of the container.
        """
        return self.__newpatch

    def collide(self, other):
        """
        Handles the collision between the container and another ball.

        Args:
            other (Ball): The ball colliding with the container.

        Return:
            None: This method does not return anything but changes 
                  the attributes of the ball and container.
        """
        v_1 = self.vel()
        v_2 = other.vel()
        normal = other.pos() / np.linalg.norm(other.pos())
        other.set_vel(v_2 - 2 * np.dot(v_2, normal) * normal)

        dp = (other.vel() - v_2)*other.mass()
        self._impulse += np.linalg.norm(dp)
        self.set_vel(v_1-dp/self.mass())

    def dp_tot(self):
        """
        Retrieves the total impulse received by the container due to collisions.

        Returns:
            float: The total impulse received.
        """
        return self._impulse
