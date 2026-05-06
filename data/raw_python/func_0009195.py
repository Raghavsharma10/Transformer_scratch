def optimize(self, angles0, target):
        """Calculate an optimum argument of an objective function."""
        def new_objective(angles):
            a = angles - angles0
            if isinstance(self.smooth_factor, (np.ndarray, list)):
                if len(a) == len(self.smooth_factor):
                    return (self.f(angles, target) +
                            np.sum(self.smooth_factor * np.power(a, 2)))
                else:
                    raise ValueError('len(smooth_factor) != number of joints')
            else:
                return (self.f(angles, target) +
                        self.smooth_factor * np.sum(np.power(a, 2)))

        return scipy.optimize.minimize(
            new_objective,
            angles0,
            **self.optimizer_opt).x