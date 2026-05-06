def Ode(self, z):
        """
        Returns the sum of :func:`~classylss.binding.Background.Omega_lambda`
        and :func:`~classylss.binding.Background.Omega_fld`.
        """
        return self.bg.Omega_lambda(z) + self.bg.Omega_fld(z)