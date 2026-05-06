def pay(self, predecessor):
        """If the predecessor is not None, gives the appropriate amount of
        payoff to the predecessor in payment for its contribution to this
        match set's expected future payoff. The predecessor argument should
        be either None or a MatchSet instance whose selected action led
        directly to this match set's situation.

        Usage:
            match_set = model.match(situation)
            match_set.pay(previous_match_set)

        Arguments:
            predecessor: The MatchSet instance which was produced by the
                same classifier set in response to the immediately
                preceding situation, or None if this is the first situation
                in the scenario.
        Return: None
        """
        assert predecessor is None or isinstance(predecessor, MatchSet)

        if predecessor is not None:
            expectation = self._algorithm.get_future_expectation(self)
            predecessor.payoff += expectation