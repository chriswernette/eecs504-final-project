import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
import eta.core.image as etai

DEBUG = False

# U = Upper, B = Bottom, L = Left, R = Right
UL = 0
BL = 1
BR = 2
UR = 3
x = 0
y = 1

def is_billboard_present(corners):
	# Return True if billboard feasible, return False otherwise
	if DEBUG:
		print("Corners: ", corners)

	if len(corners) != 4:
		return False
	tolerated = angles_within_tol(corners,horz_tol=15,vert_tol=15)
	tolerated *= edge_lens_within_tol(corners,sim_ratio=[0.5,1.75],h_v_ratio=[1.5,5])
	return tolerated


def angles_within_tol(cor, horz_tol, vert_tol):
	tolerated = True
	tolerated *= ((90 - cor_angle(cor[UL],cor[BL])) <= vert_tol)
	tolerated *= ((90 - cor_angle(cor[BR],cor[UR])) <= vert_tol)
	tolerated *= (cor_angle(cor[BL],cor[BR]) <= horz_tol)
	tolerated *= (cor_angle(cor[UR],cor[UL]) <= horz_tol)
	if not tolerated and DEBUG:
		print("Angles out of tolerance")
	return tolerated

def cor_angle(c1,c2):
	return math.degrees(math.atan2(abs(c1[y]-c2[y]), abs(c1[x]-c2[x])))

def edge_lens_within_tol(cor, sim_ratio, h_v_ratio):
	tolerated = True

	
	left   = np.linalg.norm(cor[UL] - cor[BL])
	right  = np.linalg.norm(cor[UR] - cor[BR])
	upper  = np.linalg.norm(cor[UL] - cor[UR])
	bottom = np.linalg.norm(cor[BL] - cor[BR])
	


	tolerated *= (sim_ratio[0] <= left/right)   *   (left/right <= sim_ratio[1])
	if not tolerated and DEBUG:
		print("Edges out of tolerance at point 1: ", left/right)
		return False
	tolerated *= (sim_ratio[0] <= upper/bottom) * (upper/bottom <= sim_ratio[1])
	if not tolerated and DEBUG:
		print("Edges out of tolerance at point 2: ", upper/bottom)
		return False

	tolerated *= (h_v_ratio[0] <= upper/left)   *   (upper/left <= h_v_ratio[1])
	if not tolerated and DEBUG:
		print("Edges out of tolerance at point 3: ", upper / left)
		return False
	tolerated *= (h_v_ratio[0] <= bottom/right) * (bottom/right <= h_v_ratio[1])
	if not tolerated and DEBUG:
		print("Edges out of tolerance at point 4: ", bottom/right)
		return False

	if not tolerated and DEBUG:
		print("Edges out of tolerance")
		return False

	return tolerated