def assert_page_source_contains(self, expected_value, failure_message='Expected page source to contain: "{}"'):
        """
        Asserts that the page source contains the string passed in expected_value
        """
        assertion = lambda: expected_value in self.driver_wrapper.page_source()
        self.webdriver_assert(assertion, unicode(failure_message).format(expected_value))