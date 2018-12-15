import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
import eta.core.image as etai

# U = Upper, B = Bottom, L = Left, R = Right
UL = 0
BL = 1
BR = 2
UR = 3
x = 0
y = 1

def is_billboard_present(corners):
	# Return True if billboard feasible, return False otherwise
    if len(corners) != 4:
        return False
    tolerated = angles_within_tol(corners,horz_tol=30,vert_tol=30)
    tolerated *= edge_lens_within_tol(corners,sim_ratio=[0.8,1.2],h_v_ratio=[2,3])
    return tolerated


def angles_within_tol(cor, horz_tol, vert_tol):
    tolerated = True
    tolerated *= ((90 - cor_angle(cor[UL],cor[BL])) <= vert_tol)
    tolerated *= ((90 - cor_angle(cor[BR],cor[UR])) <= vert_tol)
    tolerated *= (cor_angle(cor[BL],cor[BR]) <= horz_tol)
    tolerated *= (cor_angle(cor[UR],cor[UL]) <= horz_tol)
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
    tolerated *= (sim_ratio[0] <= upper/bottom) * (upper/bottom <= sim_ratio[1])

    tolerated *= (h_v_ratio[0] <= upper/left)   *   (upper/left <= h_v_ratio[1])
    tolerated *= (h_v_ratio[0] <= bottom/right) * (bottom/right <= h_v_ratio[1])

    return tolerated