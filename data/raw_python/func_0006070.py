def wait_until_stale(self, timeout=None):
        """
        Waits for the element to go stale in the DOM

        @type timeout:          int
        @param timeout:         override for default timeout

        @rtype:                 WebElementWrapper
        @return:                Self
        """
        timeout = timeout if timeout is not None else self.driver_wrapper.timeout

        def wait():
            """
            Wrapper to wait for element to be stale
            """
            WebDriverWait(self.driver, timeout).until(EC.staleness_of(self.element))
            return self

        return self.execute_and_handle_webelement_exceptions(wait, 'wait for staleness')