def calcTransitDuration(self, circular=False):
        """ Estimation of the primary transit time assuming a circular orbit (see :py:func:`equations.transitDuration`)
        """

        try:
            if circular:
                return eq.transitDurationCircular(self.P, self.star.R, self.R, self.a, self.i)
            else:
                return eq.TransitDuration(self.P, self.a, self.R, self.star.R, self.i, self.e, self.periastron).Td
        except (ValueError,
                AttributeError,  # caused by trying to rescale nan i.e. missing i value
                HierarchyError):  # i.e. planets that dont orbit stars
            return np.nan