def optimize(self, x0, target):
        """Calculate an optimum argument of an objective function."""
        x = x0
        for _ in range(self.maxiter):
            delta = np.linalg.solve(self.h(x, target), -self.g(x, target))
            x = x + delta
            if np.linalg.norm(delta) < self.tol:
                break
        return x