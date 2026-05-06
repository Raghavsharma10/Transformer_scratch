def param_names(self):
        """Combined names for the hyperparameters for the kernel, noise kernel and (if present) mean function.
        """
        pn = CombinedBounds(self.k.param_names, self.noise_k.param_names)
        if self.mu is not None:
            pn = CombinedBounds(pn, self.mu.param_names)
        return pn