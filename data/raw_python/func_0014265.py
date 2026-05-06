def simulate(self, steps=None):
        """
        Advance simulation n steps
        """
        if steps is None:
            steps = self.steps
        self.simulation.step(steps)