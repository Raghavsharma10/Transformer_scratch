def _dEndfr(self):
        """Eq. 4 from Orazio and Samsing (2018)

        Takes f in rest frame.

        """
        Mc = self._chirp_mass()
        return (np.pi**(2./3.)*Mc**(5./3.)/(3.*(1.+self.z)**(1./3.)
                * (self.freqs_orb/(1.+self.z))**(1./3.))*(2./self.n)**(2./3.)
                * self._g_func()/self._f_func())