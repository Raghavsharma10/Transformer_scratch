def get(self, rule, default=None):
        """Return the existing version of the given rule. If the rule is
        not present in the classifier set, return the default. If no
        default was given, use None. This is useful for eliminating
        duplicate copies of rules.

        Usage:
            unique_rule = model.get(possible_duplicate, possible_duplicate)

        Arguments:
            rule: The ClassifierRule instance which may be a duplicate of
                another already contained in the classifier set.
            default: The value returned if the rule is not a duplicate of
                another already contained in the classifier set.
        Return:
            If the rule is a duplicate of another already contained in the
            classifier set, the existing one is returned. Otherwise, the
            value of default is returned.
        """
        assert isinstance(rule, ClassifierRule)

        if (rule.condition not in self._population or
                rule.action not in self._population[rule.condition]):
            return default
        return self._population[rule.condition][rule.action]