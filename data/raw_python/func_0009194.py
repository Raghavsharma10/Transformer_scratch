def optimize(self, angles0, target):
        """Calculate an optimum argument of an objective function."""
        def new_objective(angles):
            return self.f(angles, target)

        return scipy.optimize.minimize(
            new_objective,
            angles0,
            **self.optimizer_opt).x