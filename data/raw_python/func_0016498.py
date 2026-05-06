def get_future_expectation(self, match_set):
        """Return a numerical value representing the expected future payoff
        of the previously selected action, given only the current match
        set. The match_set argument is a MatchSet instance representing the
        current match set.

        Usage:
            match_set = model.match(situation)
            expectation = model.algorithm.get_future_expectation(match_set)
            payoff = previous_reward + discount_factor * expectation
            previous_match_set.payoff = payoff

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A float, the estimate of the expected near-future payoff for
            the situation for which match_set was generated, based on the
            contents of match_set.
        """
        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self

        return self.discount_factor * (
            self.idealization_factor * match_set.best_prediction +
            (1 - self.idealization_factor) * match_set.prediction
        )