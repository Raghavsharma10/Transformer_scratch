def assert_true(self, value, failure_message='Expected value to be True, was: {}'):
        """
        Asserts that a value is true

        @type value:    bool
        @param value:   value to test for True
        """
        assertion = lambda: bool(value)
        self.webdriver_assert(assertion, unicode(failure_message).format(value))