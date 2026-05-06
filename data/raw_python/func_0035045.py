def _chirp_mass(self):
        """Chirp mass calculation

        """
        return (self.m1*self.m2)**(3./5.)/(self.m1+self.m2)**(1./5.)