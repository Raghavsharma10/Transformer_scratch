def apply_payoff(self):
        """Apply the payoff that has been accumulated from immediate
        reward and/or payments from successor match sets. Attempting to
        call this method before an action has been selected or after it
        has already been called for the same match set will result in a
        ValueError.

        Usage:
            match_set.select_action()
            match_set.payoff = reward
            match_set.apply_payoff()

        Arguments: None
        Return: None
        """
        if self._selected_action is None:
            raise ValueError("The action has not been selected yet.")
        if self._closed:
            raise ValueError("The payoff for this match set has already"
                             "been applied.")
        self._algorithm.distribute_payoff(self)
        self._payoff = 0
        self._algorithm.update(self)
        self._closed = True