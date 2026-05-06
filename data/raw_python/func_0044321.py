def _generate_initial_score(self):
        """Runs the evaluation function for the initial pose."""
        self.current_energy = self.eval_fn(self.polypeptide, *self.eval_args)
        self.best_energy = copy.deepcopy(self.current_energy)
        self.best_model = copy.deepcopy(self.polypeptide)
        return