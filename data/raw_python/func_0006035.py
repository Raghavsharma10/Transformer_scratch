def assert_not_equal(self, actual_val, unexpected_val, failure_message='Expected values to differ: "{}" and "{}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the '!=' operator
        """
        assertion = lambda: unexpected_val != actual_val
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_val, unexpected_val))