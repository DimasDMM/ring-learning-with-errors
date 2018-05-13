from math import floor, sqrt, pi
from numpy.polynomial import polynomial as p
import numpy as np
import sys

def cross_round(x, q):
	result = np.floor((4 * x) / q) % 2
	return result

def mod_round(x, q):
	result = np.around((2 * x) / q) % 2
	return result

def dbl(x, n, q):
	dist = [-1, 0, 1]
	prob = [0.25, 0.5, 0.25]
	e = np.random.choice(dist, n, p=prob)
	dbl_x = (2 * x) % (2 * q)
	dbl_x = p.polysub(dbl_x, e) % (2 * q)
	return dbl_x

def rec(w, b, q):
	n = len(w)
	r = np.zeros(n)
	E_left = -q / 4
	E_right = q / 4
	i = 0
	while i < n:
		if b[i] == 0:
			r[i] = rec_I_0(w[i], q, E_left, E_right)
		else:
			r[i] = rec_I_1(w[i], q, E_left, E_right)
		i += 1
	return r

def rec_I_0(w, q, E_left, E_right):
	I_start = 0
	I_end = round(q / 2) - 1
	bound_left = (I_start + E_left) % (2 * q)
	bound_right = (I_end + E_right) % (2 * q)
	if bound_right < bound_left:
		temp = bound_right
		bound_right = bound_left
		bound_left = temp
	if w >= bound_left and w < bound_right:
		return 0
	return 1

def rec_I_1(w, q, E_left, E_right):
	I_start = -floor(q/2)
	I_end = -1
	bound_left = (I_start + E_left) % (2 * q)
	bound_right = (I_end + E_right) % (2 * q)
	if bound_right < bound_left:
		temp = bound_right
		bound_right = bound_left
		bound_left = temp
	if w >= bound_left and w < bound_right:
		return 0
	return 1

def invert(k):
    n = len(k)
    i = 0
    while i < n:
        if k[i] == 1:
            k[i] = 0
        else:
            k[i] = 1
        i += 1
    return k


# Shared parameters between Alice and Bob: n, q, sigma, A
# Set n, q and sigma
# Generate A using a uniform distribution
n = 1024
q = 2**32 - 1
sigma = 8 / sqrt(2 * pi)
A = np.floor(np.random.random(size=(n)) * q) % q


# Alice
sA = np.floor(np.random.normal(0, scale=sigma, size=(n)))
eA = np.floor(np.random.normal(0, scale=sigma, size=(n)))

bA = p.polymul(A, sA) % q
bA = p.polyadd(bA, eA) % q


# Bob
# Gets bA from Alice
sB = np.floor(np.random.normal(0, scale=sigma, size=(n)))
eB = np.floor(np.random.normal(0, scale=sigma, size=(n)))

bB = p.polymul(A, sB) % q
bB = p.polyadd(bB, eB) % q

eeB = np.floor(np.random.normal(0, scale=sigma, size=(n)))

v = p.polymul(bA, sB) % q
v = p.polyadd(v, eeB) % q

dbl_v = dbl(v, n, q)

c = cross_round(dbl_v, 2*q)

kB = mod_round(dbl_v, 2*q)


# Alice
# Gets bB and c from Bob
recA = (2 * bB) % (2 * q)
recA = p.polymul(recA, sA) % (2 * q)
kA = rec(recA, c, q)
kA = invert(kA) # Necessary to invert all bits


# Results
print("\n-Params---")
print(" n: ", n)
print(" q: ", q)
print(" A: ", len(A), " | ", A)

print("\n-Alice---")
print(" s: ", len(sA), " | ", sA)
print(" e: ", len(eA), " | ", eA)
print(" b: ", len(bA), " | ", bA)

print("\n-Bob---")
print(" s': ", len(sB), " | ", sB)
print(" e': ", len(eB), " | ", eB)
print(" b': ", len(bB), " | ", bB)
print(" e'': ", len(eeB), " | ", eeB)

print("\n-Computed data---")
print(" v: ", len(v), " | ", v)
print(" dbl_v: ", len(dbl_v), " | ", dbl_v)
print(" c: ", len(c), " | ", c)
print(" recA: ", len(recA), " | ", recA)

print("\n-Generated Keys---")
print(" Ka: ", len(kA), " | ", kA)
print(" Kb: ", len(kB), " | ", kB)

i = 0
result = 0
while i < len(kA):
	if kA[i] == kB[i]:
		result += 1
	i += 1
result = result * 100 / len(kA)
print(" Equals: ", result, "%")

sys.exit(0)
