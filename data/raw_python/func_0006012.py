def wait_until_stale(self, locator, timeout=None):
        """
        Waits for an element to be stale in the DOM

        @type locator:  webdriverwrapper.support.locator.Locator
        @param locator: the locator or css string to search for the element
        @type timeout:  int
        @param timeout:  the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout

        def wait():
            '''
            Wait function passed to executor
            '''
            element = WebDriverWait(self.driver, timeout).until(EC.staleness_of(
                (self.locator_handler.parse_locator(locator).By, self.locator_handler.parse_locator(locator).value)))

            return WebElementWrapper.WebElementWrapper(self, locator, element)

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, locator, 'Timeout waiting for element to become stale')