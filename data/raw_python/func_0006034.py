def assert_numbers_almost_equal(self, actual_val, expected_val, allowed_delta=0.0001,
                                    failure_message='Expected numbers to be within {} of each other: "{}" and "{}"'):
        """
        Asserts that two numbers are within an allowed delta of each other
        """
        assertion = lambda: abs(expected_val - actual_val) <= allowed_delta
        self.webdriver_assert(assertion, unicode(failure_message).format(allowed_delta, actual_val, expected_val))