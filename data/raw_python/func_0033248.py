def a(self, **kwargs):
        '''
        Returns the lattice parameter, a, in Angstroms at a given
        temperature, `T`, in Kelvin (default: 300 K).
        '''
        T = kwargs.get('T', 300.)
        return (self.a_300K(**kwargs) +
                self.thermal_expansion(**kwargs) * (T - 300.))