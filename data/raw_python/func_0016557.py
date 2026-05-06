def select_action(self):
        """Select an action according to the action selection strategy of
        the associated algorithm. If an action has already been selected,
        raise a ValueError instead.

        Usage:
            if match_set.selected_action is None:
                match_set.select_action()

        Arguments: None
        Return:
            The action that was selected by the action selection strategy.
        """
        if self._selected_action is not None:
            raise ValueError("The action has already been selected.")
        strategy = self._algorithm.action_selection_strategy
        self._selected_action = strategy(self)
        return self._selected_action