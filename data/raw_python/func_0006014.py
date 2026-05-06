def wait_until_title_is(self, title, timeout=None):
        """
        Waits for title to be exactly <partial_title>

        @type title:    str
        @param title:   the exact title to locate
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
            return WebDriverWait(self.driver, timeout).until(EC.title_is(title))

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, title, 'Timeout waiting for title to be: ' + str(title))