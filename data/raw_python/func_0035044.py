def _t_of_e(self, a0=None, t_start=None, f0=None, ef=None, t_obs=5.0):
        """Rearranged versions of Peters equations

        This function calculates the semi-major axis and eccentricity over time.

        """
        if ef is None:
            ef = np.ones_like(self.e0)*0.0000001

        beta = 64.0/5.0*self.m1*self.m2*(self.m1+self.m2)

        e_vals = np.asarray([np.linspace(ef[i], self.e0[i], self.num_points)
                            for i in range(len(self.e0))])
        integrand = self._find_integrand(e_vals)
        integral = np.asarray([np.trapz(integrand[:, i:], x=e_vals[:, i:])
                              for i in range(e_vals.shape[1])]).T

        if a0 is None and f0 is None:

            a0 = (19./12.*t_start*beta*1/integral[:, 0])**(1./4.) * self._f_e(e_vals[:, -1])

        elif a0 is None:
            a0 = ((self.m1 + self.m2)/self.f0**2)**(1./3.)

        c0 = self._c0_func(a0, self.e0)

        a_vals = c0[:, np.newaxis]*self._f_e(e_vals)

        delta_t = 12./19*c0[:, np.newaxis]**4/beta[:, np.newaxis]*integral

        return e_vals, a_vals, delta_t