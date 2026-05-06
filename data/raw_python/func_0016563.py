def add(self, rule):
        """Add a new classifier rule to the classifier set. Return a list
        containing zero or more rules that were deleted from the classifier
        by the algorithm in order to make room for the new rule. The rule
        argument should be a ClassifierRule instance. The behavior of this
        method depends on whether the rule already exists in the
        classifier set. When a rule is already present, the rule's
        numerosity is added to that of the version of the rule already
        present in the population. Otherwise, the new rule is captured.
        Note that this means that for rules already present in the
        classifier set, the metadata of the existing rule is not
        overwritten by that of the one passed in as an argument.

        Usage:
            displaced_rules = model.add(rule)

        Arguments:
            rule: A ClassifierRule instance which is to be added to this
                classifier set.
        Return:
            A possibly empty list of ClassifierRule instances which were
            removed altogether from the classifier set (as opposed to
            simply having their numerosities decremented) in order to make
            room for the newly added rule.
        """

        assert isinstance(rule, ClassifierRule)

        condition = rule.condition
        action = rule.action

        # If the rule already exists in the population, then we virtually
        # add the rule by incrementing the existing rule's numerosity. This
        # prevents redundancy in the rule set. Otherwise we capture the
        # new rule.
        if condition not in self._population:
            self._population[condition] = {}

        if action in self._population[condition]:
            existing_rule = self._population[condition][action]
            existing_rule.numerosity += rule.numerosity
        else:
            self._population[condition][action] = rule

        # Any time we add a rule, we need to call this to keep the
        # population size under control.
        return self._algorithm.prune(self)