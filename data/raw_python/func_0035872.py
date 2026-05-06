def d(self):
        """ Note this should work from child parents as .d propergates, calculates using the star estimation method
        estimateDistance and estimateAbsoluteMagnitude
        """
        # TODO this will only work from a star or below. good thing?
        d = self.parent.d
        if ed_params.estimateMissingValues:
            if d is np.nan:
                d = self.estimateDistance()
                if d is not np.nan:
                    self.flags.addFlag('Estimated Distance')
            return d
        else:
            return np.nan