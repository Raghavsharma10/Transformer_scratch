def _convert_units(self):
        """Convert units to geometrized units.

        Change to G=c=1 (geometrized) units for ease in calculations.

        """
        self.m1 = self.m1*M_sun*ct.G/ct.c**2
        self.m2 = self.m2*M_sun*ct.G/ct.c**2
        initial_cond_type_conversion = {
            'time': ct.c*ct.Julian_year,
            'frequency': 1./ct.c,
            'separation': ct.parsec,
        }

        self.initial_point = self.initial_point*initial_cond_type_conversion[self.initial_cond_type]

        self.t_obs = self.t_obs*ct.c*ct.Julian_year
        return