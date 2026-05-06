def _g_func(self):
        """Eq. 20 in Peters and Mathews 1963.

        """
        return (self.n**4./32.
                * ((jv(self.n-2., self.n*self.e_vals)
                   - 2. * self.e_vals*jv(self.n-1., self.n*self.e_vals)
                   + 2./self.n * jv(self.n, self.n*self.e_vals)
                   + 2.*self.e_vals*jv(self.n+1., self.n*self.e_vals)
                   - jv(self.n+2., self.n*self.e_vals))**2.
                   + (1.-self.e_vals**2.) * (jv(self.n-2., self.n*self.e_vals)
                   - 2.*jv(self.n, self.n*self.e_vals)
                   + jv(self.n+2., self.n*self.e_vals))**2.
                   + 4./(3.*self.n**2.)*(jv(self.n, self.n*self.e_vals))**2.))