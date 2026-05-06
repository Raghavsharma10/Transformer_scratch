def _apex2qd_nonvectorized(self, alat, alon, height):
        """Convert from apex to quasi-dipole (not-vectorised)

        Parameters
        -----------
        alat : (float)
            Apex latitude in degrees
        alon : (float)
            Apex longitude in degrees
        height : (float)
            Height in km

        Returns
        ---------
        qlat : (float)
            Quasi-dipole latitude in degrees
        qlon : (float)
            Quasi-diplole longitude in degrees
        """

        alat = helpers.checklat(alat, name='alat')

        # convert modified apex to quasi-dipole:
        qlon = alon

        # apex height
        hA = self.get_apex(alat)

        if hA < height:
            if np.isclose(hA, height, rtol=0, atol=1e-5):
                # allow for values that are close
                hA = height
            else:
                estr = 'height {:.3g} is > apex height '.format(np.max(height))
                estr += '{:.3g} for alat {:.3g}'.format(hA, alat)
                raise ApexHeightError(estr)

        qlat = np.sign(alat) * np.degrees(np.arccos(np.sqrt((self.RE + height) /
                                                            (self.RE + hA))))

        return qlat, qlon