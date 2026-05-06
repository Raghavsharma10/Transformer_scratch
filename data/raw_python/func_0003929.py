def update(self, gradient, step):
        """Update the search direction given the latest gradient and step"""
        do_sd = self.gradient_old is None
        self.gradient_old = self.gradient
        self.gradient = gradient
        if do_sd:
            self._update_sd()
        else:
            self._update_cg()