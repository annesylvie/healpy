# 
#  This file is part of Healpy.
# 
#  Healpy is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
# 
#  Healpy is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with Healpy; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
# 
#  For more information about Healpy, see http://code.google.com/p/healpy
#
"""
========================================================
polarizationplot.py : Healpix polarization drawing tools
========================================================

This module provides tools for plotting polarization maps U and Q.

"""
import numpy as np
import math as M
import healpy as hp
import pylab as pl

def compute_vecs_pol(x, y, z, q, u,factor):
 """Define all we need to compute the polarization points which will be used to draw polarization on sphere.
 Parameters
 ----------
 x, y, z : cartesian coordinates of the central point of the pixel
 q, u : Stokes parameters at this point
 factor : mutliplicative factor for the length of the polarization drawn on the map
 """
 n_hat = np.array([x,y,z])
 rho = M.sqrt(x**2+y**2)
 theta_hat=np.array([x*z/rho,y*z/rho,-rho])
 phi_hat  =np.array([-y,x,0])/rho
 def delta_a(q,u):
   p_mat=np.array([[q,-u],[-u,-q]])/M.sqrt(u**2+q**2)+np.eye(2)
   e_one=np.array([0.,1.])
   v=np.dot(p_mat,e_one)
   nv=M.sqrt(np.dot(v,v))
   if nv < 1.e-2 :
     e_two=np.array([1.,0.])
     v=np.dot(p_mat,e_two)
     nv=M.sqrt(np.dot(v,v))
   v=v/nv
   v=M.sqrt(u**2+q**2)*v
   return(v)
 a_vec=delta_a(q,u)
 n_plus=n_hat+factor*(a_vec[0]*theta_hat+a_vec[1]*phi_hat)
 n_minus=n_hat-factor*(a_vec[0]*theta_hat+a_vec[1]*phi_hat)
 return((n_plus,n_minus,n_hat))

def pol_points(x, y, z, q, u,factor,npoints=10):
 """Here, given some pixel point, we compute a set of points on the sphere which are the projection of polarization vectors on the sphere. These points will be used to draw polarization lines."""
 n_plus,n_minus,n_hat=compute_vecs_pol(x, y, z, q, u,factor)
 x_pts=np.linspace(n_plus[0],n_minus[0],num=npoints)
 y_pts=np.linspace(n_plus[1],n_minus[1],num=npoints)
 z_pts=np.linspace(n_plus[2],n_minus[2],num=npoints)
 points=[]
 for i in range(npoints):
   n=np.array([x_pts[i],y_pts[i],z_pts[i]])
   n=n/M.sqrt(np.dot(n,n))
   points.append(n)
 return(points)

def draw_pol(map_q,map_u,nside, ampl=1):
 """Draw polarization field on current map.

 Parameters
 ----------
 map_q : map of the Q Stokes parameter
 map_u : map of the U Stokes parameter
 nside : nside of the displayed map
 ampl : mutliplicative factor for the length of the polarization line drawn on the map
 """
 #f = pl.gcf()
 nb_pixels= 12* (nside**2)
 factor = ampl*1/(np.sqrt(12)*nside)
 map_q_grade = hp.ud_grade(map_q, nside)
 map_u_grade = hp.ud_grade(map_u, nside)
 for i in range(nb_pixels):
   x_pix,y_pix,z_pix = hp.pix2vec(nside,i)
   points = pol_points(x_pix,y_pix,z_pix,map_q_grade[i],map_u_grade[i],factor)
   theta = []
   phi = []
   for point in points:
     t,p = hp.vec2ang(point)
     theta.append(t[0])
     phi.append(p[0])
   hp.projplot(theta, phi)

def plot_local_meridian(nside, ipix):
 theta0, phi0 = hp.pix2ang(nside, ipix)
 theta = theta0+np.linspace(-0.1,0.1,100)
 phi = phi0+np.zeros(100)
 hp.projplot(theta,phi)


