def scalefactor(self, other, qmin=None, qmax=None, Npoints=None):
        """Calculate a scaling factor, by which this curve is to be multiplied to best fit the other one.

        Inputs:
            other: the other curve (an instance of GeneralCurve or of a subclass of it)
            qmin: lower cut-off (None to determine the common range automatically)
            qmax: upper cut-off (None to determine the common range automatically)
            Npoints: number of points to use in the common x-range (None defaults to the lowest value among
                the two datasets)

        Outputs:
            The scaling factor determined by interpolating both datasets to the same abscissa and calculating
                the ratio of their integrals, calculated by the trapezoid formula. Error propagation is
                taken into account.
        """
        if qmin is None:
            qmin = max(self.q.min(), other.q.min())
        if qmax is None:
            xmax = min(self.q.max(), other.q.max())
        data1 = self.trim(qmin, qmax)
        data2 = other.trim(qmin, qmax)
        if Npoints is None:
            Npoints = min(len(data1), len(data2))
        commonx = np.linspace(
                max(data1.q.min(), data2.q.min()), min(data2.q.max(), data1.q.max()), Npoints)
        data1 = data1.interpolate(commonx)
        data2 = data2.interpolate(commonx)
        return nonlinear_odr(data1.Intensity, data2.Intensity, data1.Error, data2.Error, lambda x, a: a * x, [1])[0]