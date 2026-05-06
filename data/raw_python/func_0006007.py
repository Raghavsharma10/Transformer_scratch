def wait_until(self, wait_function, failure_message=None, timeout=None):
        """
        Base wait method: called by other wait functions to execute wait

        @type wait_function:    types.FunctionType
        @param wait_function:   Generic function to be executed
        @type failure_message:  str
        @param failure_message: Message to fail with if exception is raised
        @type timeout:          int
        @param timeout:         timeout override

        @rtype:                 webdriverwrapper.WebElementWrapper
        @return:                Returns the element found
        """
        timeout = timeout if timeout is not None else self.timeout
        failure_message = failure_message if failure_message is not None else \
            'Timeout waiting for custom function to return True'

        def wait():
            '''
            Wait function passed to executor
            '''
            return WebDriverWait(self, timeout).until(lambda dw: wait_function())

        return self.execute_and_handle_webdriver_exceptions(wait, timeout, None, failure_message)