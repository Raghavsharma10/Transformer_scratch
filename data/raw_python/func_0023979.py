def free_params(self):
        """Combined free hyperparameters for the kernel, noise kernel and (if present) mean function.
        """
        p = CombinedBounds(self.k.free_params, self.noise_k.free_params)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.free_params)
        return p