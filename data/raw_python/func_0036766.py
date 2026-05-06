def Onu(self, z):
        """
        Returns the sum of :func:`~classylss.binding.Background.Omega_ncdm`
        and :func:`~classylss.binding.Background.Omega_ur`.
        """
        return self.bg.Omega_ncdm(z) + self.bg.Omega_ur(z)