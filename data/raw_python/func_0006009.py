def wait_until_not_present(self, locator, timeout=None):
        """
        Waits for an element to no longer be present

        @type locator:  webdriverwrapper.support.locator.Locator
        @param locator: the locator or css string to search for the element
        @type timeout:  int
        @param timeout:  the maximum number of seconds the driver will wait before timing out

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        # TODO: rethink about neg case with is_present and waiting too long
        timeout = timeout if timeout is not None else self.timeout
        this = self  # for passing WebDriverWrapperReference to WebDriverWait

        def wait():
            '''
            Wait function pasted to executor
            '''
            return WebDriverWait(self.driver, timeout).until(lambda d: not this.is_present(locator))

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, locator, 'Timeout waiting for element not to be present')