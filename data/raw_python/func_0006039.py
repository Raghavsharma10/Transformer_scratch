def assert_not_in(self, actual_collection_or_string, unexpected_value,
                      failure_message='Expected "{1}" not to be in "{0}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the 'not in' operator
        """
        assertion = lambda: unexpected_value not in actual_collection_or_string
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_collection_or_string, unexpected_value))