def T(self):
        """ Looks for the temperature in the catalogue, if absent it calculates it using calcTemperature()

        :return: planet temperature
        """
        paramTemp = self.getParam('temperature')

        if not paramTemp is np.nan:
            return paramTemp
        elif ed_params.estimateMissingValues:
            self.flags.addFlag('Calculated Temperature')
            return self.calcTemperature()
        else:
            return np.nan