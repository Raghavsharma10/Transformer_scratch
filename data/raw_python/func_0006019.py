def wait_until_jquery_requests_are_closed(self, timeout=None):
        """Waits for AJAX requests made through

        @type timeout:     int
        @param timeout:    the maximum number of seconds the driver will wait before timing out
        @return: None
        """
        timeout = timeout if timeout is not None else self.timeout

        def wait():
            '''
            Wait function passed to executor
            '''
            WebDriverWait(self.driver, timeout).until(
                lambda d: self.js_executor.execute_template('isJqueryAjaxComplete', {}))
            return True

        return self.execute_and_handle_webdriver_exceptions(
            wait, timeout, None, 'Timeout waiting for all jQuery AJAX requests to close')