def best_prediction(self):
        """The highest value from among the predictions made by the action
        sets in this match set."""
        if self._best_prediction is None and self._action_sets:
            self._best_prediction = max(
                action_set.prediction
                for action_set in self._action_sets.values()
            )
        return self._best_prediction