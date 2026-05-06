def wait_until_alert_is_present(self, timeout=None):
        """
        Waits for an alert to be present

        @type timeout:          int
        @param timeout:         the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout
        locator = None

        def wait():
            '''
            Wait function passed to executor
            '''
            return WebDriverWait(self.driver, timeout).until(EC.alert_is_present())

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, locator, 'Timeout waiting for alert to be present')