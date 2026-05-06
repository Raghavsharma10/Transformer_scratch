def _set_payoff(self, payoff):
        """Setter method for the payoff property."""
        if self._selected_action is None:
            raise ValueError("The action has not been selected yet.")
        if self._closed:
            raise ValueError("The payoff for this match set has already"
                             "been applied.")
        self._payoff = float(payoff)