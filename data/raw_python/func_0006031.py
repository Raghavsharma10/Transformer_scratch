def webdriver_assert(self, assertion, failure_message='Failed Assertion'):
        """
        Assert the assertion, but throw a WebDriverAssertionException if assertion fails
        """
        try:
            assert assertion() is True
        except AssertionError:
            raise WebDriverAssertionException.WebDriverAssertionException(self.driver_wrapper, failure_message)

        return True