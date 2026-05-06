def wait_until_page_source_contains(self, text, timeout=None):
        """
        Waits for the page source to contain <text>

        @type text:             str
        @param text:            the text to search for
        @type timeout:          int
        @param timeout:         the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout

        def wait():
            '''
            Wait function passed to executor
            '''
            WebDriverWait(self.driver, timeout).until(lambda d: text in self.page_source())
            return self.page_source()

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, text, 'Timeout waiting for source to contain: {}'.format(text))