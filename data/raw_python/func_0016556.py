def best_actions(self):
        """A tuple containing the actions whose action sets have the best
        prediction."""
        if self._best_actions is None:
            best_prediction = self.best_prediction
            self._best_actions = tuple(
                action
                for action, action_set in self._action_sets.items()
                if action_set.prediction == best_prediction
            )
        return self._best_actions