def assert_equals(self, actual_val, expected_val, failure_message='Expected values to be equal: "{}" and "{}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the '==' operator
        """
        assertion = lambda: expected_val == actual_val
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_val, expected_val))