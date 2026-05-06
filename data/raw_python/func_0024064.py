def free_params(self, value):
        """Set the free parameters. Note that this bypasses enforce_bounds.
        """
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.free_params = value[:self.k.num_free_params]
        self.w.free_params = value[self.k.num_free_params:self.k.num_free_params + self.w.num_free_params]