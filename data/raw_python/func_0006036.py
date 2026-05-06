def assert_is(self, actual_val, expected_type, failure_message='Expected type to be "{1}," but was "{0}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the 'is' operator
        """
        assertion = lambda: expected_type is actual_val
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_val, expected_type))