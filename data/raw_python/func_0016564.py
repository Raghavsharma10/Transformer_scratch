def discard(self, rule, count=1):
        """Remove one or more instances of a rule from the classifier set.
        Return a Boolean indicating whether the rule's numerosity dropped
        to zero. (If the rule's numerosity was already zero, do nothing and
        return False.)

        Usage:
            if rule in model and model.discard(rule, count=3):
                print("Rule numerosity dropped to zero.")

        Arguments:
            rule: A ClassifierRule instance whose numerosity is to be
                decremented.
            count: An int, the size of the decrement to the rule's
                numerosity; default is 1.
        Return:
            A bool indicating whether the rule was removed altogether from
            the classifier set, as opposed to simply having its numerosity
            decremented.
        """
        assert isinstance(rule, ClassifierRule)
        assert isinstance(count, int) and count >= 0

        rule = self.get(rule)
        if rule is None:
            return False

        # Only actually remove the rule if its numerosity drops below 1.
        rule.numerosity -= count
        if rule.numerosity <= 0:
            # Ensure that if there is still a reference to this rule
            # elsewhere, its numerosity is still well-defined.
            rule.numerosity = 0

            del self._population[rule.condition][rule.action]
            if not self._population[rule.condition]:
                del self._population[rule.condition]
            return True

        return False