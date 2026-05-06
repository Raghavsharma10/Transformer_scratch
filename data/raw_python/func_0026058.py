def log_interp1d(self, xx, yy, kind='linear'):
        """
        Performs a log space 1d interpolation.
        :param xx: the x values.
        :param yy: the y values.
        :param kind: the type of interpolation to apply (as per scipy interp1d)
        :return: the interpolation function.
        """
        logx = np.log10(xx)
        logy = np.log10(yy)
        lin_interp = interp1d(logx, logy, kind=kind)
        log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
        return log_interp