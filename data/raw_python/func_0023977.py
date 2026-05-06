def params(self):
        """Combined hyperparameters for the kernel, noise kernel and (if present) mean function.
        """
        p = CombinedBounds(self.k.params, self.noise_k.params)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.params)
        return p