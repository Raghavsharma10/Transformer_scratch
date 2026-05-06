def prepare(self, f):
        """Accept an objective function for optimization."""
        self.g = autograd.grad(f)
        self.h = autograd.hessian(f)