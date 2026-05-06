def assert_subset(self, subset, superset, failure_message='Expected collection "{}" to be a subset of "{}'):
        """
        Asserts that a superset contains all elements of a subset
        """
        assertion = lambda: set(subset).issubset(set(superset))
        failure_message = unicode(failure_message).format(superset, subset)
        self.webdriver_assert(assertion, failure_message)