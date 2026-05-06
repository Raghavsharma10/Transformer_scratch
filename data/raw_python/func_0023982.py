def free_param_names(self):
        """Combined free hyperparameter names for the kernel, noise kernel and (if present) mean function.
        """
        p = CombinedBounds(self.k.free_param_names, self.noise_k.free_param_names)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.free_param_names)
        return p