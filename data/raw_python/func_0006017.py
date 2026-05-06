def wait_until_text_is_not_empty(self, locator, timeout=None):
        """
        Waits for an element's text to not be empty

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used to find element
        @type timeout:          int
        @param timeout:         the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout

        self.wait_for(locator) # first check that element exists

        def wait():
            '''
            Wait function passed to executor
            '''
            WebDriverWait(self.driver, timeout).until(lambda d: len(self.find(locator).text()) > 0)
            return self.find(locator)

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, locator, 'Timeout waiting for element to contain some text')