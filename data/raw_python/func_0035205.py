def clearFixedEffect(self):
        """ erase all fixed effects """
        self.A = []
        self.F = []
        self.F_any = np.zeros((self.N,0))
        self.clear_cache()