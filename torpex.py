#!/usr/bin/env python3
"""
Create a BOUT++ grid for TORPEX from an input file giving coil currents and positions

Input file should contain coil parameters, for each coil:
    R: major radius in metres
    Z: major radius in metres
    I: clockwise current in Amps

Note: positions of cell corners are generated first, grid points are then put in the
centre of the cell.
"""

plotStuff = True

import numpy
import warnings
from mesh import Mesh, MeshContour, Point2D
from equilibrium import Equilibrium
if plotStuff:
    from matplotlib import pyplot

class TORPEXMagneticField(Equilibrium):
    """
    Magnetic configuration defined by coil positions and currents for the TORPEX device
    """

    # TORPEX wall is a circle radius 0.2 m around (1 m, 0 m)
    awall = 0.2
    Rcentre = 1.
    Zcentre = 0.

    # Bounding box
    Rmin = 0.8
    Rmax = 1.2
    Zmin = -.2
    Zmax = .2

    def __init__(self, coils, Bt_axis):
        self.coils = coils
        self.magneticFunctions()

        # TORPEX plasma pressure so low fpol is constant
        self.fpol = lambda psi: Bt_axis / self.Rcentre

        try:
            self.x_points = [self.findSaddlePoint(self.Rmin, self.Rmax, 0.8*self.Zmin,
                                                  0.8*self.Zmax)]
            self.psi_sep = self.psi(*self.x_points[0])
        except:
            warnings.warn('Warning: failed to find X-point. Equilibrium generation will '
                    'fail')

    def TORPEX_wall(self, theta):
        """
        Return the location of the TORPEX wall parameterized by the angle theta
        anticlockwise around the centre of the vacuum vessel
        """
        return Point2D(self.Rcentre + self.awall*numpy.cos(theta),
                       self.Zcentre + self.awall*numpy.sin(theta))

    def addWallToPlot(self, npoints=100):
        theta = numpy.linspace(0., 2.*numpy.pi, npoints+1)
        pyplot.plot(*self.TORPEX_wall(theta))

    def magneticFunctions(self):
        """
        Calculate toroidal (anticlockwise) component of magnetic vector potential due to coils
        See for example http://physics.usask.ca/~hirose/p812/notes/Ch3.pdf

        Note e_R x e_phi = e_Z
        """
        import sympy
        from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
        import scipy.special

        R,Z = sympy.symbols('R Z')
        mu0 = 4.e-7*sympy.pi

        potential = 0*R

        for coil in self.coils:
            # little-r is the vector position from the centre of the coil to (R,Z)
            # sinTheta is the angle between r and the axis through the centre of the coil
            rSquared = R**2 + (Z - coil.Z)**2
            r = sympy.sqrt(rSquared)
            sinTheta = R / r
            kSquared = 4*coil.R*r*sinTheta / (rSquared + coil.R**2 + 2*coil.R*r*sinTheta)
            potential += -(
              coil.I*coil.R / sympy.sqrt(r**2 + coil.R**2 + 2*coil.R*r*sinTheta) / kSquared
              * ( (2-kSquared)*elliptic_k(kSquared) - 2*elliptic_e(kSquared) )
              )

        # multiply by costant pre-factor
        potential *= mu0/sympy.pi

        dAdR = sympy.diff(potential, R)
        dAdZ = sympy.diff(potential, Z)
        modGradASquared = dAdR**2 + dAdZ**2

        self.psi = sympy.lambdify([R,Z], potential, modules=['numpy',
            {'elliptic_k':scipy.special.ellipk, 'elliptic_e':scipy.special.ellipe}])
        self.f_R = sympy.lambdify([R,Z], dAdR/modGradASquared, modules=['numpy',
            {'elliptic_k':scipy.special.ellipk, 'elliptic_e':scipy.special.ellipe}])
        self.f_Z = sympy.lambdify([R,Z], dAdZ/modGradASquared, modules=['numpy',
            {'elliptic_k':scipy.special.ellipk, 'elliptic_e':scipy.special.ellipe}])
        self.Bp_R = sympy.lambdify([R,Z], dAdZ, modules=['numpy',
            {'elliptic_k':scipy.special.ellipk, 'elliptic_e':scipy.special.ellipe}])
        self.Bp_Z = sympy.lambdify([R,Z], -1/R * sympy.diff(R*potential, R), modules=['numpy',
            {'elliptic_k':scipy.special.ellipk, 'elliptic_e':scipy.special.ellipe}])

    def findSeparatrix(self, atol = 2.e-8, npoints=100):
        """
        Find the separatrix to grid from.

        For TORPEX, follow 4 legs away from the x-point, starting with a rough guess and
        then refining to the separatrix value of A_toroidal.
        """
        wall_position = lambda s: self.TORPEX_wall(s*2.*numpy.pi)

        assert len(self.x_points) == 1 # should be one X-point for TORPEX configuration
        xpoint = self.x_points[0]

        boundary = self.findRoots_1d(lambda s: self.psi(*wall_position(s)) - self.psi_sep,
                                4, 0., 1.)

        # put lower left leg first in list, go clockwise
        boundary = boundary[2::-1] + [boundary[3]]

        boundaryPoints = tuple(wall_position(s) for s in boundary)

        legs = []
        s = numpy.linspace(10.*atol, 1., npoints)
        for point in boundaryPoints:
            legR = xpoint.R + s*(point.R - xpoint.R)
            legZ = xpoint.Z + s*(point.Z - xpoint.Z)
            leg = MeshContour([Point2D(R,Z) for R,Z in zip(legR, legZ)], self.psi,
                              self.psi_sep)
            leg = leg.getRefined(atol=atol, width=0.2)
            legs.append(leg)

        # note legs are ordered in theta
        self.separatrix = {'inner_lower_divertor': legs[0],
                           'inner_upper_divertor': legs[1],
                           'outer_upper_divertor': legs[2],
                           'outer_lower_divertor': legs[3]}

def parseInput(filename):
    import yaml
    from collections import namedtuple

    with open(filename, 'r') as inputfile:
        coil_inputs, mesh_inputs = yaml.safe_load_all(inputfile)
    print('Coils:',coil_inputs['Coils'])
    
    Coil = namedtuple('Coil', 'R, Z, I')
    return [Coil(**c) for c in coil_inputs['Coils']], coil_inputs['Bt_axis'], mesh_inputs['Mesh']

def createMesh(filename):
    # parse input file
    coils, Bt_axis, meshOptions = parseInput(filename)

    equilibrium = TORPEXMagneticField(coils, Bt_axis)

    print('X-point',equilibrium.x_points[0],'with A_toroidal='+str(equilibrium.psi_sep))

    equilibrium.findSeparatrix()

    return Mesh(equilibrium, meshOptions)

if __name__ == '__main__':
    from sys import argv, exit

    filename = argv[1]
    gridname = 'torpex.grd.nc'

    mesh = createMesh(filename)

    mesh.geometry()

    if plotStuff:
        mesh.equilibrium.plotPotential()
        mesh.equilibrium.addWallToPlot()
        pyplot.plot(*mesh.equilibrium.x_points[0], 'rx')
        mesh.plotPoints(xlow=True, ylow=True, corners=True)
        pyplot.show()

    mesh.writeGridfile(gridname)

    exit(0)
