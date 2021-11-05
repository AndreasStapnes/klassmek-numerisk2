from __future__ import annotations
from typing import NamedTuple, overload, Union
import numpy as np
from methods import RK4
from methods import Phase as CartesianPhase
from methods import SYS

m_1_global = 2e3
m_2_global = 15
l_1_global = 1.2
l_2_global = 5.7
l_3_global = 3.2
l_4_global = 5
g = 9.8


def a_mat(theta: float, theta_prime: float, psi: float, psi_prime: float,
          m_1: float = m_1_global, m_2: float = m_2_global, l_1: float = l_1_global, l_2: float = l_2_global,
          l_3: float = l_3_global, l_4: float = l_4_global) -> np.ndarray:
    return np.array([
        -l_1**2*m_1 - l_2**2*m_2 + 2*l_2*l_4*m_2*np.cos(psi) - l_4**2*m_2,
        -l_4*m_2*(l_2*np.cos(psi)-l_4),
        -l_4*m_2*(l_2*np.cos(psi)-l_4),
        -l_4**2*m_2
    ]).reshape((2, 2))



def a_inverse(theta: float, theta_prime: float, psi: float, psi_prime: float,
          m_1: float = m_1_global, m_2: float = m_2_global, l_1: float = l_1_global, l_2: float = l_2_global,
          l_3: float = l_3_global, l_4: float = l_4_global):
    return np.linalg.inv(a_mat(theta, theta_prime, psi, psi_prime, m_1, m_2, l_1, l_2, l_3, l_4))

def b_vec(theta: float, theta_prime: float, psi: float, psi_prime: float,
          m_1: float = m_1_global, m_2: float = m_2_global, l_1: float = l_1_global, l_2: float = l_2_global,
          l_3: float = l_3_global, l_4: float = l_4_global):
    return np.array([
        g*m_1*l_1*np.sin(theta) - g*m_2*(l_2*np.sin(theta) + l_4*np.sin(psi-theta)) - l_2*l_4*m_2*(psi_prime-2*theta_prime)*np.sin(psi)*psi_prime,
        l_4*m_2*(g*np.sin(psi-theta)-l_2*np.sin(psi)*theta_prime**2)
    ])


def function_for_RK(phase: Phase, t:float):
    theta = phase.theta
    psi = phase.psi
    theta_prime = phase.theta_prime
    psi_prime = phase.psi_prime
    theta_double_prime, psi_double_prime = a_inverse(theta, theta_prime, psi, psi_prime) @ b_vec(theta, theta_prime, psi, psi_prime)
    return Phase(theta_prime, theta_double_prime, psi_prime, psi_double_prime)




class Phase(NamedTuple):
    theta: float
    theta_prime: float
    psi: float
    psi_prime: float

    dt: Union[float, None] = None

    @overload
    def __add__(self, iterations: int): ...
    @overload
    def __add__(self, rhs: Phase): ...

    def __add__(self, rhs: Union[int, Phase]):
        if type(rhs) is int:
            calculated_phase = self
            for enum in range(rhs):
                calculated_phase, t = calculated_phase.make_propagated(self.dt)
            return calculated_phase
        elif type(rhs) is Phase:
            if self.dt is not None:
                dt = self.dt
                if rhs.dt is not None:
                    if rhs.dt != self.dt: raise Exception("The phases have non-matching dt-values!")
            elif rhs.dt is not None:
                dt = rhs.dt
            else:
                dt = None
            return Phase(self.theta + rhs.theta,
                         self.theta_prime + rhs.theta_prime,
                         self.psi + rhs.psi,
                         self.psi_prime + rhs.psi_prime,
                         dt)
        else:
            raise Exception("Invalid addition of states")

    def make_propagated(self, timestep: float = None):
        if timestep is None: timestep = self.dt
        return RK4(function_for_RK, self, 0, timestep)

    def __mul__(self, rhs:float):
        return Phase(self.theta * rhs, self.theta_prime*rhs,
                     self.psi*rhs, self.psi_prime*rhs, self.dt)

    def __rmul__(self, lhs:float):
        return self*lhs

    def __truediv__(self, rhs:float):
        return self*(1/rhs)

    def m1_velocity(self):
        return l_1_global*np.array([np.cos(self.theta), np.sin(self.theta)])

    def m2_velocity(self):
        return np.array([-l_2_global*np.cos(self.theta)*self.theta_prime - l_4_global*np.cos(self.psi-self.theta)*(self.psi_prime - self.theta_prime),
                         -l_2_global*np.sin(self.theta)*self.theta_prime + l_4_global*np.sin(self.psi-self.theta)*(self.psi_prime-self.theta_prime)])

    def m1_r(self):
        return np.array([l_1_global*np.sin(self.theta), -l_1_global*np.cos(self.theta) + l_3_global])

    def m2_r(self):
        return np.array([-l_2_global*np.sin(self.theta)-l_4_global*np.cos(self.psi-self.theta),
                         l_2_global*np.cos(self.theta) + l_3_global - l_4_global*np.cos(self.psi-self.theta)])

    def m2_cartesian_phase(self):
        return CartesianPhase(*self.m2_r(), *self.m2_velocity())

    def propagate_until_release(self, release_angle: float) -> Phase:
        local_phase = self
        while local_phase.theta > release_angle:
            #print(local_phase)
            local_phase = local_phase + 1
        return local_phase




import matplotlib.pyplot as plt

def T(theta: float, theta_prime: float, psi: float, psi_prime: float, *args):
    m_1 = m_1_global;
    m_2 = m_2_global
    l_1 = l_1_global;
    l_2 = l_2_global;
    l_3 = l_3_global;
    l_4 = l_4_global
    return 1/2*(m_1*l_1**2 + m_2*l_2**2)*theta_prime**2 + 1/2*m_2*l_4**2*(psi_prime-theta_prime)**2 + m_2*l_2*l_4*theta_prime*(psi_prime-theta_prime)*np.cos(psi)

def V(theta: float, theta_prime: float, psi: float, psi_prime: float, *args):
    m_1 = m_1_global;
    m_2 = m_2_global
    l_1 = l_1_global;
    l_2 = l_2_global;
    l_3 = l_3_global;
    l_4 = l_4_global
    return -l_1*m_1*np.cos(theta)*g + m_1*l_3*g + m_2*l_2*np.cos(theta)*g + m_2*l_3*g - m_2*l_4*np.cos(psi-theta)*g

def H(theta: float, theta_prime: float, psi: float, psi_prime: float, *args):
    m_1 = m_1_global; m_2 = m_2_global
    l_1 = l_1_global; l_2 = l_2_global; l_3 = l_3_global; l_4 = l_4_global
    L = lambda theta, theta_prime, psi, psi_prime: T(theta, theta_prime, psi, psi_prime) - V(theta, theta_prime, psi, psi_prime)
    p_theta = (L(theta, theta_prime+1e-4, psi, psi_prime)-L(theta, theta_prime, psi, psi_prime))*1e4
    p_psi = (L(theta, theta_prime, psi, psi_prime+1e-4)-L(theta, theta_prime, psi, psi_prime))*1e4
    return p_theta*theta_prime + p_psi*psi_prime - L(theta, theta_prime, psi, psi_prime)
