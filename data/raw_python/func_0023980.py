def free_params(self, value):
        """Set the free parameters. Note that this bypasses enforce_bounds.
        """
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.free_params = value[:self.k.num_free_params]
        self.noise_k.free_params = value[self.k.num_free_params:self.k.num_free_params + self.noise_k.num_free_params]
        if self.mu is not None:
            self.mu.free_params = value[self.k.num_free_params + self.noise_k.num_free_params:]