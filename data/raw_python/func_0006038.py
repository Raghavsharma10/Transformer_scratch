def assert_in(self, actual_collection_or_string, expected_value, failure_message='Expected "{1}" to be in "{0}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the 'in' operator
        """
        assertion = lambda: expected_value in actual_collection_or_string
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_collection_or_string, expected_value))