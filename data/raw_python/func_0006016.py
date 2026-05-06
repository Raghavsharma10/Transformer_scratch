def wait_until_text_contains(self, locator, text, timeout=None):
        """
        Waits for an element's text to contain <text>

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used to find element
        @type text:             str
        @param text:            the text to search for
        @type timeout:          int
        @param timeout:         the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout
        this = self

        self.wait_for(locator) # first check that element exists

        def wait():
            '''
            Wait function passed to executor
            '''
            WebDriverWait(self.driver, timeout).until(lambda d: text in this.find(locator).text())
            return this.find(locator)

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, locator, 'Timeout waiting for text to contain: ' + str(text))