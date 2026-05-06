def minimize(self, tolerance=None, max_iterations=None):
        """
        Minimize energy of the system until meeting `tolerance` or
        performing `max_iterations`.
        """
        if tolerance is None:
            tolerance = self.minimization_tolerance
        if max_iterations is None:
            max_iterations = self.minimization_max_iterations
        self.simulation.minimizeEnergy(tolerance * u.kilojoules_per_mole, max_iterations)