def getLimbdarkeningCoeff(self, wavelength=1.22):  # TODO replace with pylightcurve
        """ Looks up quadratic limb darkening parameter from the star based on T, logg and metalicity.

        :param wavelength: microns
        :type wavelength: float

        :return: limb darkening coefficients 1 and 2
        """
        # TODO check this returns correct value - im not certain
        # The intervals of values in the tables
        tempind = [ 3500., 3750., 4000., 4250., 4500., 4750., 5000., 5250., 5500., 5750., 6000., 6250.,
                 6500., 6750., 7000., 7250., 7500., 7750., 8000., 8250., 8500., 8750., 9000., 9250.,
                 9500., 9750., 10000., 10250., 10500., 10750., 11000., 11250., 11500., 11750., 12000., 12250.,
                 12500., 12750., 13000., 14000., 15000., 16000., 17000., 19000., 20000., 21000., 22000., 23000.,
                 24000., 25000., 26000., 27000., 28000., 29000., 30000., 31000., 32000., 33000., 34000., 35000.,
                 36000., 37000., 38000., 39000., 40000., 41000., 42000., 43000., 44000., 45000., 46000., 47000.,
                 48000., 49000., 50000.]
        lggind = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
        mhind = [-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.5, 1.]

        # Choose the values in the table nearest our parameters
        tempselect = _findNearest(tempind, float(self.T))
        lgselect = _findNearest(lggind, float(self.calcLogg()))
        mhselect = _findNearest(mhind, float(self.Z))

        quadratic_filepath = resource_stream(__name__, 'data/quadratic.dat')
        coeffTable = np.loadtxt(quadratic_filepath)

        foundValues = False
        for i in range(len(coeffTable)):
            if coeffTable[i, 2] == lgselect and coeffTable[i, 3] == tempselect and coeffTable[i, 4] == mhselect:
                if coeffTable[i, 0] == 1:
                    u1array = coeffTable[i, 8:]  # Limb darkening parameter u1 for each wl in waveind
                    u2array = coeffTable[i+1, 8:]
                    foundValues = True
                    break

        if not foundValues:
            raise ValueError('No limb darkening values could be found')  # TODO replace with better exception

        waveind = [0.365, 0.445, 0.551, 0.658, 0.806, 1.22, 1.63, 2.19, 3.45]  # Wavelengths available in table

        # Interpolates the value at wavelength from values in the table (waveind)
        u1AtWavelength = np.interp(wavelength, waveind, u1array, left=0, right=0)
        u2AtWavelength = np.interp(wavelength, waveind, u2array, left=0, right=0)

        return u1AtWavelength, u2AtWavelength