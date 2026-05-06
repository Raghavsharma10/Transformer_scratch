def cover(self, match_set):
        """Return a new classifier rule that can be added to the match set,
        with a condition that matches the situation of the match set and an
        action selected to avoid duplication of the actions already
        contained therein. The match_set argument is a MatchSet instance
        representing the match set to which the returned rule may be added.

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
            A new ClassifierRule instance, appropriate for the addition to
            match_set and to the classifier set from which match_set was
            drawn.
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.model.algorithm is self

        # Create a new condition that matches the situation.
        condition = bitstrings.BitCondition.cover(
            match_set.situation,
            self.wildcard_probability
        )

        # Pick a random action that (preferably) isn't already suggested by
        # some other rule for this situation.
        action_candidates = (
            frozenset(match_set.model.possible_actions) -
            frozenset(match_set)
        )
        if not action_candidates:
            action_candidates = match_set.model.possible_actions
        action = random.choice(list(action_candidates))

        # Create the new rule.
        return XCSClassifierRule(
            condition,
            action,
            self,
            match_set.time_stamp
        )