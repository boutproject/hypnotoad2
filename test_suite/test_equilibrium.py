import numpy
import pytest
from copy import deepcopy
from equilibrium import *
from test_utils import *

class TestPoints:
    p0 = Point2D(1., 2.)
    p1 = Point2D(3., 4.)

    def test_add(self):
        p = self.p0 + self.p1
        assert p.R == tight_approx(4.)
        assert p.Z == tight_approx(6.)

    def test_sub(self):
        p = self.p0 - self.p1
        assert p.R == tight_approx(-2.)
        assert p.Z == tight_approx(-2.)

    def test_mul(self):
        p = self.p0 * 1.5
        assert p.R == tight_approx(1.5)
        assert p.Z == tight_approx(3.)

    def test_rmul(self):
        p = 1.5 * self.p0
        assert p.R == tight_approx(1.5)
        assert p.Z == tight_approx(3.)

    def test_div(self):
        p = self.p0 / 3.
        assert p.R == tight_approx(1./3.)
        assert p.Z == tight_approx(2./3.)

    def test_iter(self):
        assert [x for x in self.p0] == tight_approx([1.,2.])

    def test_repr(self):
        assert str(self.p0) == "Point2D(1.0,2.0)"

    def test_distance(self):
        assert calc_distance(self.p0, self.p1) == tight_approx(2.*numpy.sqrt(2.))

class TestContour:
    @pytest.fixture
    def testcontour(self):
        psifunc = lambda R,Z: R**2 + Z**4
        psi_xpoint = 0.

        # make a circle, not centred on origin
        class returnObject:
            npoints = 23
            r = 1.
            R0 = .2
            Z0 = .3
            theta = numpy.linspace(0.,2.*numpy.pi,npoints)
            R = R0 + r*numpy.cos(theta)
            Z = Z0 + r*numpy.sin(theta)
            c = PsiContour([Point2D(R,Z) for R,Z in zip(R,Z)], psifunc, psi_xpoint)

        return returnObject()

    def test_distance(self, testcontour):
        segment_length = 2.*testcontour.r*numpy.sin(2.*numpy.pi/(testcontour.npoints-1)/2.)
        assert testcontour.c.distance == tight_approx(segment_length*numpy.arange(23))


    def test_iter(self, testcontour):
        clist = list(testcontour.c)

        for i,item in enumerate(clist):
            assert item.R == tight_approx(testcontour.R[i])
            assert item.Z == tight_approx(testcontour.Z[i])

    def test_getitem(self, testcontour):
        p = testcontour.c[5]
        assert p.R == tight_approx(testcontour.R[5])
        assert p.Z == tight_approx(testcontour.Z[5])

    def test_append(self, testcontour):
        c = testcontour.c
        expected_distance = c.distance[-1] + numpy.sqrt((1.-c[-1].R)**2 + (1.-c[-1].Z)**2)
        c.append(Point2D(1.,1.))
        assert c.distance[-1] == tight_approx(expected_distance)

    def test_reverse(self, testcontour):
        c = testcontour.c
        orig = deepcopy(c)

        c.reverse()

        n = len(orig)
        total_d = orig.distance[-1]
        for i in range(n):
            assert orig[n-1-i].R == tight_approx(c[i].R)
            assert orig[n-1-i].Z == tight_approx(c[i].Z)
            assert total_d - orig.distance[n-1-i] == tight_approx(c.distance[i])

    def test_refine(self, testcontour):
        # PsiContour.refine just calls PsiContour.getRefined, so this tests both

        c = testcontour.c
        c.psival = .7
        # c does not start close to a contour of psi_func, so need to use a large width
        c.refine(width=2., atol=1.e-13)
        for p in c:
            assert c.psi(p.R, p.Z) == tight_approx(.7)

    def test_interpFunction(self, testcontour):
        f = testcontour.c.interpFunction()
        p = f(0.5*testcontour.c.distance[-1])
        assert p.R == tight_approx(testcontour.R0 - testcontour.r)
        assert p.Z == tight_approx(testcontour.Z0)

    def test_getRegridded(self):
        # make a circular contour in a circular psi
        psifunc = lambda R,Z: R**2 + Z**2

        npoints = 1000
        r = 1.
        theta = numpy.linspace(0., 2.*numpy.pi, npoints)
        R = r*numpy.cos(theta)
        Z = r*numpy.sin(theta)
        orig = PsiContour([Point2D(R,Z) for R,Z in zip(R,Z)], psifunc, 1.)

        newNpoints = 97
        sfunc = lambda i: numpy.sqrt(i / (newNpoints - 1)) * 2. * numpy.pi * r
        newTheta = sfunc(numpy.arange(newNpoints))
        newR = r*numpy.cos(newTheta)
        newZ = r*numpy.sin(newTheta)

        new = orig.getRegridded(newNpoints, sfunc=sfunc, width=1.e-3)

        assert [p.R for p in new] == pytest.approx(newR, abs=2.e-5)
        assert [p.Z for p in new] == pytest.approx(newZ, abs=2.e-5)

    def test_getRegridded_extend(self, testcontour):
        # make a circular contour in a circular psi
        psifunc = lambda R,Z: R**2 + Z**2

        npoints = 23
        r = 1.
        theta = numpy.linspace(0., 2.*numpy.pi, npoints)
        R = r*numpy.cos(theta)
        Z = r*numpy.sin(theta)
        orig = PsiContour([Point2D(R,Z) for R,Z in zip(R,Z)], psifunc, 1.)

        new = orig.getRegridded(testcontour.npoints, width=.1, extend_lower=1, extend_upper=2)

        assert numpy.array([[*p] for p in new[1:-2]]) == tight_approx(numpy.array([[*p] for p in orig]))

        # test the extend_lower
        assert [*new[0]] == pytest.approx([*orig[-2]], abs=2.e-3)

        # test the extend_upper
        assert [*new[-2]] == pytest.approx([*orig[1]], abs=2.e-3)
        assert [*new[-1]] == pytest.approx([*orig[2]], abs=2.e-2)

class TestEquilibrium:

    @pytest.fixture
    def eq(self):
        eq = Equilibrium()
        return eq

    def test_make1dGrid(self, eq):
        n = 4
        f = lambda i: i**2
        r = eq.make1dGrid(n, f)
        assert r == tight_approx([0., 0.5, 1., 2.5, 4., 6.5, 9., 12.5, 16.])

    def test_getSqrtPoloidalDistanceFuncLinear(self, eq):
        L = 2.
        N = 10.
        N_norm = 1
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(10.) == tight_approx(2.)
        # f(i) = i/N*L
        assert f(3.) == tight_approx(0.6)

    def test_getSqrtPoloidalDistanceFuncDLower(self, eq):
        d_lower = 0.01
        L = 2.
        N = 10.
        N_norm = 40.
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_lower = d_lower)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for i<<1, f ~ 2*d_lower*sqrt(i/N_norm) + d_lower*i/N_norm
        itest = 0.01
        assert f(itest) == pytest.approx(2.*d_lower*numpy.sqrt(itest/N_norm)
                                         + d_lower*itest/N_norm, abs=1.e-5)

    def test_getSqrtPoloidalDistanceFuncDUpper(self, eq):
        d_upper = 0.01
        L = 2.
        N = 10.
        N_norm = 40.
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_upper = d_upper)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for (N-i)<<1, f ~ L - 2*d_upper*sqrt((N-i)/N_norm) - d_upper*(N-i)/N_norm
        itest = N - 0.01
        assert f(itest) == pytest.approx(L - 2.*d_upper*numpy.sqrt((N - itest)/N_norm)
                                         - d_upper*(N - itest)/N_norm, abs=1.e-5)

    def test_getSqrtPoloidalDistanceFuncDBoth(self, eq):
        d_lower = 0.1
        d_upper = 0.2
        L = 2.
        N = 10.
        N_norm = 40.
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_lower = d_lower, d_upper = d_upper)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for i<<1, f ~ 2*d_lower*sqrt(i/N_norm) + d_lower*i/N_norm
        itest = 0.01
        assert f(itest) == pytest.approx(2.*d_lower*numpy.sqrt(itest/N_norm)
                                         + d_lower*itest/N_norm, abs=1.e-5)
        # for (N-i)<<1, f ~ L - 2*d_upper*sqrt((N-i)/N_norm) - d_upper*(N-i)/N_norm
        itest = N - 0.01
        assert f(itest) == pytest.approx(L - 2.*d_upper*numpy.sqrt((N - itest)/N_norm)
                                         - d_upper*(N - itest)/N_norm, abs=1.e-5)

    def test_getSqrtPoloidalDistanceFuncDLower2(self, eq):
        d_lower = 0.01
        d_sqrt_lower = 0.05
        L = 2.
        N = 10.
        N_norm = 2.
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_lower = d_lower, d_sqrt_lower =
                d_sqrt_lower)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for i<<1, f ~ 2*d_sqrt_lower*sqrt(i/N_norm) + d_lower*i/N_norm
        itest = 0.01
        assert f(itest) == pytest.approx(2.*d_sqrt_lower*numpy.sqrt(itest/N_norm)
                                         + d_lower*itest/N_norm, abs=1.e-5)

    def test_getSqrtPoloidalDistanceFuncDUpper2(self, eq):
        d_upper = 0.01
        d_sqrt_upper = 0.05
        L = 2.
        N = 10.
        N_norm = 1
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_upper = d_upper, d_sqrt_upper = d_sqrt_upper)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for (N-i)<<1, f ~ L - 2*d_sqrt_upper*sqrt((N-i)/N_norm) - d_upper*(N-i)/N_norm
        itest = N - 0.01
        assert f(itest) == pytest.approx(L - 2.*d_sqrt_upper*numpy.sqrt((N - itest)/N_norm)
                                         - d_upper*(N - itest)/N_norm, abs=1.e-5)

    def test_getSqrtPoloidalDistanceFuncDBoth2(self, eq):
        d_lower = 0.01
        d_sqrt_lower = 0.05
        d_upper = 0.2
        d_sqrt_upper = 0.07
        L = 2.
        L = 2.
        N = 10.
        N_norm = 1
        f = eq.getSqrtPoloidalDistanceFunc(L, N, N_norm, d_lower = d_lower,
                d_sqrt_lower = d_sqrt_lower, d_upper = d_upper,
                d_sqrt_upper = d_sqrt_upper)
        # f(0) = 0
        assert f(0.) == tight_approx(0.)
        # f(N) = L
        assert f(N) == tight_approx(L)
        # for i<<1, f ~ 2*d_sqrt_lower*sqrt(i/N_norm) + d_lower*i/N_norm
        itest = 0.01
        assert f(itest) == pytest.approx(2.*d_sqrt_lower*numpy.sqrt(itest/N_norm)
                                         + d_lower*itest/N_norm, abs=1.e-5)
        # for (N-i)<<1, f ~ L - 2*d_sqrt_upper*sqrt((N-i)/N_norm) - d_upper*(N-i)/N_norm
        itest = N - 0.01
        assert f(itest) == pytest.approx(L - 2.*d_sqrt_upper*numpy.sqrt((N - itest)/N_norm)
                                         - d_upper*(N - itest)/N_norm, abs=1.e-5)