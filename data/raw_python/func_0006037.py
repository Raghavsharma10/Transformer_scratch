def assert_is_not(self, actual_val, unexpected_type,
                      failure_message='Expected type not to be "{1}," but was "{0}"'):
        """
        Calls smart_assert, but creates its own assertion closure using
        the expected and provided values with the 'is not' operator
        """
        assertion = lambda: unexpected_type is not actual_val
        self.webdriver_assert(assertion, unicode(failure_message).format(actual_val, unexpected_type))