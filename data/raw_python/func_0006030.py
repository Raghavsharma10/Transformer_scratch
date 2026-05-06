def spin_assert(self, assertion, failure_message='Failed Assertion', timeout=None):
        """
        Asserts that assertion function passed to it will return True,
        trying every 'step' seconds until 'timeout' seconds have passed.
        """
        timeout = self.timeout if timeout is None else timeout
        time_spent = 0
        while time_spent < timeout:
            try:
                assert assertion() is True
                return True
            except AssertionError:
                pass
            sleep(self.step)
            time_spent += 1
        raise WebDriverAssertionException.WebDriverAssertionException(self.driver_wrapper, failure_message)