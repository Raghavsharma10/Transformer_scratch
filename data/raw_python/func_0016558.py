def _set_selected_action(self, action):
        """Setter method for the selected_action property."""
        assert action in self._action_sets

        if self._selected_action is not None:
            raise ValueError("The action has already been selected.")
        self._selected_action = action