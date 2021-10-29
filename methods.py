from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


#Constants and parameters:

v_0 = 700  # firing velocity in m/s
theta = 30  # firing angle in degrees

a = 6.5e-3  # constant in adiabatic approximation?
T0 = 288
y_0 = 1e4  # k_B*T/mg
alfa = 2.5  # constant for air?
drag_coefficient = 4e-5  # per air particle mass, in m^-1#!/usr/bin/python
g = 9.81

y0 = 1e4

# Sistnevnte er en bash-kommando for aa kjore resten av filen i python
# om filen kjores direkte fra kommandolinjen
from typing import Callable, Tuple, NamedTuple, Union, List


def RK4(f: Callable[[any,float], float], y: any, t: float, dt: float) -> Tuple[any, float]:
	k_1 = f(y, t)
	k_2 = f(y+k_1*dt/2, t+dt/2)
	k_3 = f(y+k_2*dt/2, t+dt/2)
	k_4 = f(y+k_3*dt, t+dt)
	y_next = y + 1/6*(k_1+2*k_2+2*k_3+k_4)*dt
	return y_next, t+dt


def Euler(f: Callable[[any, float], float], y:any, t:float, dt:float) -> Tuple[any, float]:
	k_1 = f(y, t)
	y_next = y + k_1*dt
	return y_next, t+dt


def crossing(ph_1: Phase, ph_2: Phase, y_crossing: float) -> Union[float, bool]:
	if not ((ph_1.y < y_crossing) ^ (ph_2.y < y_crossing)):
		return False
	r = -ph_1.y/ph_2.y
	return (r*ph_2.x + ph_1.x)/(r+1)


	

class Phase(NamedTuple):
	x: float
	y: float
	v_x: float
	v_y: float

	def speed(self):
		return np.sqrt(self.v_x**2+self.v_y**2)

	def __add__(self, rhs: Phase):
		return Phase(*[l+r for l,r in zip(self, rhs)])

	def __radd__(self, lhs: Phase):
		return self + lhs

	def __mul__(self, rhs:float):
		return Phase(*[l*rhs for l in self])

	def __rmul__(self, lhs:float):
		return self*lhs

	def __truediv__(self, rhs:float):
		return self*(1/rhs)


class Drag_type(Enum):
	NODRAG = "No drag"
	UNIFORM = "Uniform drag"
	ISOTHERMAL = "Isothermal drag"
	ADIABATIC = "Adiabatic drag"



def phase_derivative_nodrag(ph: Phase, t: float):
	'''Basic derivative-equation'''
	return Phase(ph.v_x, ph.v_y, 0, -g)

def phase_derivative_drag(ph: Phase, t: float, drag_multiplier=1):
	'''Derivative-equation including drag'''
	#Friction points in negative-v-direction with size B*v^2, where
	# B/m = drag_coefficient, so the acceleration caused
	#  by friction is given by drag_coefficient * v * (-v-vector)
	drag_factor = drag_coefficient*ph.speed()*drag_multiplier  #
	return Phase(ph.v_x, ph.v_y, -drag_factor*ph.v_x, -drag_factor*ph.v_y-g)

def phase_derivative_isothermal_drag(ph: Phase, t: float):
	drag_multiplier = np.exp(-ph.y/y0)
	return phase_derivative_drag(ph, t, drag_multiplier)

def phase_derivative_uniform_drag(ph: Phase, t: float):
	return phase_derivative_drag(ph, t)

def phase_derivative_adiabatic_drag(ph: Phase, t: float):
	drag_multiplier = (1-6.5e-3*ph.y/T0)**alfa
	return phase_derivative_drag(ph, t, drag_multiplier)

def phase_prime_method(drag_type:Drag_type=Drag_type.NODRAG):
	if drag_type is Drag_type.NODRAG: return phase_derivative_nodrag
	elif drag_type is Drag_type.UNIFORM: return phase_derivative_uniform_drag
	elif drag_type is Drag_type.ISOTHERMAL: return phase_derivative_isothermal_drag
	elif drag_type is Drag_type.ADIABATIC: return phase_derivative_adiabatic_drag
	else:
		raise Exception("How did you get here?")


class SYS:
	phases: List[Phase]
	crash_distance: Union[None, float]
	def __init__(self, init_phase:Phase, init_time:float=0,
				 drag_type:Drag_type=Drag_type.NODRAG):
		self.phases = [init_phase]
		self.times = [init_time]
		self.crash_distance = None

		self.prop_method = phase_prime_method(drag_type)

	def propagate_until_crash(self, y_floor:float, dt=1):
		while True:
			current_phase = self.phases[-1]
			t = self.times[-1]
			next_phase, t_next = RK4(self.prop_method, current_phase, t, dt)
			if (x_cross := crossing(current_phase, next_phase, y_floor)) is False:
				self.phases.append(next_phase)
				self.times.append(t_next)
				continue
			self.crash_distance = x_cross
			break


def polar_to_cartesian(abs_val: float, theta: float):
	return abs_val*np.cos(theta), abs_val*np.sin(theta)

