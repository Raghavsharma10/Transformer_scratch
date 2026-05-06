def covering_is_required(self, match_set):
        """Return a Boolean indicating whether covering is required for the
        current match set. The match_set argument is a MatchSet instance
        representing the current match set before covering is applied.

        Usage:
            match_set = model.match(situation)
            if model.algorithm.covering_is_required(match_set):
                new_rule = model.algorithm.cover(match_set)
                assert new_rule.condition(situation)
                model.add(new_rule)
                match_set = model.match(situation)

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A bool indicating whether match_set contains too few matching
            classifier rules and therefore needs to be augmented with a
            new one.
        """
        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self

        if self.minimum_actions is None:
            return len(match_set) < len(match_set.model.possible_actions)
        else:
            return len(match_set) < self.minimum_actions