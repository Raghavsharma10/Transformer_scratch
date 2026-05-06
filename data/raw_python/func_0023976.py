def fixed_params(self):
        """Combined fixed hyperparameter flags for the kernel, noise kernel and (if present) mean function.
        """
        fp = CombinedBounds(self.k.fixed_params, self.noise_k.fixed_params)
        if self.mu is not None:
            fp = CombinedBounds(fp, self.mu.fixed_params)
        return fp