from __future__ import annotations
from typing import NamedTuple, overload, Union
import numpy as np
from methods import RK4

m_1_global = 1e3
m_2_global = 4
l_1_global = 1
l_2_global = 14
l_3_global = 18
l_4_global = 10
g = 9.8


def a_mat(theta: float, theta_prime: float, psi: float, psi_prime: float,
          m_1: float = m_1_global, m_2: float = m_2_global, l_1: float = l_1_global, l_2: float = l_2_global,
          l_3: float = l_3_global, l_4: float = l_4_global) -> np.ndarray:
    return np.array([
        -l_1**2*m_2 - l_2**2*m_2 + 2*l_2*l_4*m_2*np.cos(psi) - l_4**2*m_2,
        -l_4*m_2*(l_2*np.cos(psi)-l_4),
        -l_4*m_2*(l_2*np.cos(psi)-l_4),
        l_4**2*m_2
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
        l_2*l_4*m_2*(psi_prime-2*theta_prime)*np.sin(psi)*psi_prime
    ])


def function_for_RK(phase: Phase, t:float):
    theta = phase.theta
    psi = phase.psi
    theta_prime = phase.theta_prime
    psi_prime = phase.psi_prime
    theta_double_prime, psi_double_prime = a_inverse(theta, theta_prime, psi, psi_prime) @ b_vec(theta, theta_prime, psi, psi_prime)
    return Phase(theta_prime, psi_prime, theta_double_prime, psi_double_prime)




class Phase(NamedTuple):
    theta: float
    psi: float
    theta_prime: float
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
                         self.psi + rhs.psi,
                         self.theta_prime + rhs.theta_prime,
                         self.psi_prime + rhs.psi_prime,
                         dt)
        else:
            raise Exception("Invalid addition of states")

    def make_propagated(self, timestep: float = None):
        if timestep is None: timestep = self.dt
        return RK4(function_for_RK, self, 0, self.dt)

    def __mul__(self, rhs:float):
        return Phase(self.theta * rhs, self.psi*rhs,
                     self.theta_prime*rhs, self.psi_prime*rhs, self.dt)

    def __rmul__(self, lhs:float):
        return self*lhs

    def __truediv__(self, rhs:float):
        return self*(1/rhs)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    phase = Phase(np.pi/2, np.pi/2, 0, 0, 1e-2)
    phases = []
    ts = np.arange(0,1000)*1e-3
    for i in range(1000):
        phases.append(phase)
        phase = phase + 1
    vals = np.array(phases).T[1]
    plt.plot(ts, vals)
    plt.show()
