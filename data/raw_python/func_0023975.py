def hyperprior(self):
        """Combined hyperprior for the kernel, noise kernel and (if present) mean function.
        """
        hp = self.k.hyperprior * self.noise_k.hyperprior
        if self.mu is not None:
            hp *= self.mu.hyperprior
        return hp