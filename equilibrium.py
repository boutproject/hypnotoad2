"""
Functions for analysing an equilibrium for which an interpolating function is given for
the potential.
"""

import numpy
from scipy.optimize import minimize_scalar, brentq, root
import warnings
from mesh import Point2D, MeshContour, calc_distance

class Equilibrium:
    """
    Base class to provide an interface to an interpolating function for the flux function psi that defines
    the magnetic equilibrium, along with some useful methods.

    psi is the magnetic flux function.

    f_R and f_Z are the components of a vector Grad(psi)/|Grad(psi)|**2. This vector
    points along a path perpendicular to psi-contours, and its value is ds/dpsi where s is
    the coordinate along the path, so we can follow the path by integrating this vector:
    R(psi) = \int_0^\psi f_R
    and
    Z(psi) = \int_0^\psi f_Z

    Derived classes must provide:
      - self.psi: function which takes two arguments, {R,Z}, and returns the value of psi
        at that position.
      - self.f_R: function which takes two arguments, {R,Z}, and returns the R
        component of the vector Grad(psi)/|Grad(psi)|**2.
      - self.f_Z: function which takes two arguments, {R,Z}, and returns the Z
        component of the vector Grad(psi)/|Grad(psi)|**2.
      - self.Bp_R: function which takes two arguments, {R,Z}, and returns the R
        component of the poloidal magnetic field.
      - self.Bp_Z: function which takes two arguments, {R,Z}, and returns the Z
        component of the poloidal magnetic field.
      - self.x_points: list of Point2D objects giving the position of the X-points
      - self.psi_sep: value of psi on the primary separatrix
      - self.fpol: poloidal current function, takes one argument, psi, and returns fpol
        (function such that B_toroidal = fpol/R)
      - self.Rmin, self.Rmax, self.Zmin, self.Zmax: positions of the corners of a bounding
        box for the gridding
    """
    def findMinimum_1d(self, pos1, pos2, atol=1.e-14):
        coords = lambda s: pos1 + s*(pos2-pos1)
        result = minimize_scalar(lambda s: self.psi(*coords(s)), method='bounded', bounds=(0., 1.), options={'xatol':atol})
        if result.success:
            return coords(result.x)
        else:
            raise ValueError('findMinimum_1d failed')

    def findMaximum_1d(self, pos1, pos2, atol=1.e-14):
        coords = lambda s: pos1 + s*(pos2-pos1)
        # minimize -f to find maximum
        result = minimize_scalar(lambda s: -self.psi(*coords(s)), method='bounded', bounds=(0., 1.), options={'xatol':atol})
        if result.success:
            return coords(result.x)
        else:
            raise ValueError('findMaximum_1d failed')

    def findExtremum_1d(self, pos1, pos2, rtol=1.e-5, atol=1.e-14):
        smallDistance = 10.*rtol*calc_distance(pos1, pos2)

        minpos = self.findMinimum_1d(pos1, pos2, atol)
        if calc_distance(pos1,minpos) > smallDistance and calc_distance(pos2,minpos) > smallDistance:
            # minimum is not at either end of the interval
            return minpos, True

        maxpos = self.findMaximum_1d(pos1, pos2, atol)
        if calc_distance(pos1,maxpos) > smallDistance and calc_distance(pos2,maxpos) > smallDistance:
            return maxpos, False

        raise ValueError("Neither minimum nor maximum found in interval")

    def findSaddlePoint(self, Rmin, Rmax, Zmin, Zmax, atol=2.e-8):
        """
        Find a saddle point in the function self.psi atol is the tolerance on the position
        of the saddle point {Rmin,Rmax}, {Zmin,Zmax} are the bounding values of the box to
        search for a saddle point in.
        """

        posTop, minTop = self.findExtremum_1d(Point2D(Rmin, Zmax), Point2D(Rmax, Zmax))
        posBottom, minBottom = self.findExtremum_1d(Point2D(Rmin, Zmin), Point2D(Rmax, Zmin))
        posLeft, minLeft = self.findExtremum_1d(Point2D(Rmin, Zmin), Point2D(Rmin, Zmax))
        posRight, minRight = self.findExtremum_1d(Point2D(Rmax, Zmin), Point2D(Rmax, Zmax))

        assert minTop == minBottom
        assert minLeft == minRight
        assert minTop != minLeft

        if minTop:
            vertSearch = self.findMaximum_1d
        else:
            vertSearch = self.findMinimum_1d

        if minLeft:
            horizSearch = self.findMaximum_1d
        else:
            horizSearch = self.findMinimum_1d

        extremumVert = Point2D(Rmin, Zmin)
        extremumHoriz = Point2D(Rmax, Zmax)

        count = 0
        while calc_distance(extremumVert, extremumHoriz) > atol:
            count = count+1

            extremumVert = vertSearch(posBottom, posTop, 0.5*atol)
            posLeft.Z = extremumVert.Z
            posRight.Z = extremumVert.Z

            extremumHoriz = horizSearch(posLeft, posRight, 0.5*atol)
            posBottom.R = extremumHoriz.R
            posTop.R = extremumHoriz.R

        print('findSaddlePoint took',count,'iterations to converge')

        return (extremumVert+extremumHoriz)/2.

    def findRoots_1d(self, f, n, xmin, xmax, atol = 2.e-8, rtol = 1.e-5, maxintervals=1024):
        """
        Find n roots of a scalar function f(x) in the range xmin<=x<=xmax
        Assume they're not too close to each other - exclude a small region around each found
        root when searching for more.
        """
        smallDistance = rtol * (xmax - xmin)
        foundRoots = 0
        roots = []
        n_intervals = n
        while True:
            interval_points = numpy.linspace(xmin, xmax, n_intervals+1)
            interval_f = f(interval_points)
            lucky_roots = numpy.where(interval_f == 0.)
            if len(lucky_roots[0]) > 0:
                raise NotImplementedError("Don't handle interval points that happen to land "
                        "on a root yet!")
            intervals_with_roots = numpy.where(numpy.sign(interval_f[:-1]) !=
                                               numpy.sign(interval_f[1:]))[0]
            if len(intervals_with_roots) >= n:
                break
            n_intervals *= 2
            if n_intervals > maxintervals:
                raise ValueError("Could not find", n, "roots when checking", maxintervals,
                                 "intervals")

        # find roots in the intervals
        for i in intervals_with_roots:
            root, info = brentq(f, interval_points[i], interval_points[i+1], xtol=atol,
                    full_output=True)
            if not info.converged:
                raise ValueError("Root finding failed in {" + str(interval_points[i]) + "," +
                        str(interval_points[i+1]) + "} with end values {" + str(interval_f[i])
                        + "," + str(interval_f[i+1]))
            roots.append(root)
            foundRoots += 1

        if foundRoots > n:
            warnings.warn('Warning: found',foundRoots,'roots but expected only',n)

        return roots

    def plotPotential(self, Rmin=None, Rmax=None, Zmin=None, Zmax=None, npoints=100,
            ncontours=40):
        from matplotlib import pyplot

        if Rmin is None: Rmin = self.Rmin
        if Rmax is None: Rmax = self.Rmax
        if Zmin is None: Zmin = self.Zmin
        if Zmax is None: Zmax = self.Zmax

        R = numpy.linspace(Rmin, Rmax, npoints)
        Z = numpy.linspace(Zmin, Zmax, npoints)
        contours = pyplot.contour(
                R, Z, self.psi(R[:,numpy.newaxis], Z[numpy.newaxis,:]).T, ncontours)
        pyplot.clabel(contours, inline=False, fmt='%1.3g')
        pyplot.axes().set_aspect('equal')