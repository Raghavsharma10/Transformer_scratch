def _hcn_func(self):
        """Eq. 56 from Barack and Cutler 2004

        """
        self.hc = 1./(np.pi*self.dist)*np.sqrt(2.*self._dEndfr())
        return