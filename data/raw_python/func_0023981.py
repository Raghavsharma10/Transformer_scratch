def free_param_bounds(self):
        """Combined free hyperparameter bounds for the kernel, noise kernel and (if present) mean function.
        """
        fpb = CombinedBounds(self.k.free_param_bounds, self.noise_k.free_param_bounds)
        if self.mu is not None:
            fpb = CombinedBounds(fpb, self.mu.free_param_bounds)
        return fpb