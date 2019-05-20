from scipy.fftpack import dct
import numpy

class DCT_2D:
    """
    Helper class to calculate the discrete cosine transform (DCT) of a 2d array, and provide
    an interpolation.

    From the scipy docuentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
    the DCT we are using is (the default, 'type II'):
    y[k] = 2* sum[n=0..N-1] x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N,
    which we apply first to columns (axis=1) and then to rows (axis=0).
    The inverse transform is (DCT 'type III' divided by 2N):
    y[k] = 1/(2N) * ( x[0] + 2 * sum[n=0..N-1] x[n]*cos(pi*(k+0.5)*n/N) ), 0 <= k < N
         = 1/N * ( x[0]/2 + sum[n=0..N-1] x[n]*cos(pi*(k+0.5)*n/N) ), 0 <= k < N
    """
    def __init__(self, Rarray, Zarray, psiRZ):
        self.Rarray = Rarray
        self.Zarray = Zarray
        self.nR = len(self.Rarray)
        self.nZ = len(self.Zarray)

        # Assume constant spacing in R and Z
        assert all(numpy.abs(self.Rarray - numpy.linspace(self.Rarray[0], self.Rarray[-1], self.nR)) < 1.e-13)
        assert all(numpy.abs(self.Zarray - numpy.linspace(self.Zarray[0], self.Zarray[-1], self.nZ)) < 1.e-13)

        self.dR = self.Rarray[1] - self.Rarray[0]
        self.dZ = self.Zarray[1] - self.Zarray[0]

        self.Rmin = self.Rarray[0]
        self.Rsize = self.Rarray[-1] - self.Rarray[0]
        self.Zmin = self.Zarray[0]
        self.Zsize = self.Zarray[-1] - self.Zarray[0]

        # Check array sizes are compatible
        assert self.nR == psiRZ.shape[1]
        assert self.nZ == psiRZ.shape[0]

        self.psiDCT = dct( dct(psiRZ, axis=0), axis=1 )

        # divide through by nR*nZ to simplify evaluation at a point
        self.psiDCT = self.psiDCT / (self.nR*self.nZ)
        # divide zero components by 2 as zero component of input should be 1/2, but will
        # get cos(0)=1 in __call__.
        self.psiDCT[0, :] /= 2.
        self.psiDCT[:, 0] /= 2.

        # save coefficients for evaluating the cosine function
        self.coef_R = (numpy.pi*numpy.arange(self.nR)/self.nR)[numpy.newaxis, :]
        self.coef_Z = (numpy.pi*numpy.arange(self.nZ)/self.nZ)[:, numpy.newaxis]

    def __call__(self, R, Z):
        R = numpy.array(R)
        Z = numpy.array(Z)

        # check inputs are compatible
        assert len(R.shape) == len(Z.shape)

        # calculate values in index space
        iR = (R - self.Rmin) / self.Rsize * (self.nR - 1)
        iZ = (Z - self.Zmin) / self.Zsize * (self.nZ - 1)

        # Using numpy array broadcasting instead of this explicit iteration requires a
        # high-dimensional array to be contlucted before the sum is calculated - this can
        # use a lot of memory [(size of R,Z)*(size of DCT), rather than either
        # (size of R,Z) or (size of DCT)]
        with numpy.nditer([iR, iZ, None]) as it:
            for ir, iz, result in it:
                result[...] = numpy.sum(self.psiDCT
                              * numpy.cos(self.coef_R*(ir + 0.5))
                              * numpy.cos(self.coef_Z*(iz + 0.5)))
            result = it.operands[2]

        return result

    def ddR(self, R, Z):
        R = numpy.array(R)
        Z = numpy.array(Z)

        # check inputs are compatible
        assert len(R.shape) == len(Z.shape)

        # calculate values in index space
        iR = (R - self.Rmin) / self.Rsize * (self.nR - 1)
        iZ = (Z - self.Zmin) / self.Zsize * (self.nZ - 1)

        # Using numpy array broadcasting instead of this explicit iteration requires a
        # high-dimensional array to be contlucted before the sum is calculated - this can
        # use a lot of memory [(size of R,Z)*(size of DCT), rather than either
        # (size of R,Z) or (size of DCT)]
        with numpy.nditer([iR, iZ, None]) as it:
            for ir, iz, result in it:
                result[...] = -numpy.sum(self.psiDCT
                              * self.coef_R/self.dR*numpy.sin(self.coef_R*(ir + 0.5))
                              * numpy.cos(self.coef_Z*(iz + 0.5)))
            result = it.operands[2]

        return result

    def ddZ(self, R, Z):
        R = numpy.array(R)
        Z = numpy.array(Z)

        # check inputs are compatible
        assert len(R.shape) == len(Z.shape)

        # calculate values in index space
        iR = (R - self.Rmin) / self.Rsize * (self.nR - 1)
        iZ = (Z - self.Zmin) / self.Zsize * (self.nZ - 1)

        # Using numpy array broadcasting instead of this explicit iteration requires a
        # high-dimensional array to be contlucted before the sum is calculated - this can
        # use a lot of memory [(size of R,Z)*(size of DCT), rather than either
        # (size of R,Z) or (size of DCT)]
        with numpy.nditer([iR, iZ, None]) as it:
            for ir, iz, result in it:
                result[...] = -numpy.sum(self.psiDCT
                              * numpy.cos(self.coef_R*(ir + 0.5))
                              * self.coef_Z/self.dZ*numpy.sin(self.coef_Z*(iz + 0.5)))
            result = it.operands[2]

        return result
