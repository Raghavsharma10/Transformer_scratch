def compute(a, b):
        """
        Compute an optimal displacement between two ndarrays.

        Finds the displacement between two ndimensional arrays. Arrays must be
        of the same size. Algorithm uses a cross correlation, computed efficiently
        through an n-dimensional fft.

        Parameters
        ----------
        a : ndarray
            The first array

        b : ndarray
            The second array
        """
        from numpy.fft import rfftn, irfftn
        from numpy import unravel_index, argmax

        # compute real-valued cross-correlation in fourier domain
        s = a.shape
        f = rfftn(a)
        f *= rfftn(b).conjugate()
        c = abs(irfftn(f, s))

        # find location of maximum
        inds = unravel_index(argmax(c), s)

        # fix displacements that are greater than half the total size
        pairs = zip(inds, a.shape)
        # cast to basic python int for serialization
        adjusted = [int(d - n) if d > n // 2 else int(d) for (d, n) in pairs]

        return Displacement(adjusted)