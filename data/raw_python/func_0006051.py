def set(self, val, force_set=False):
        """
        Sets an input with a specified value; if force_set=True, will set through javascript if webdriver fails
        NOTE: if val is None, this function will interpret this to be an empty string

        @type val:          str
        @param val:         string to send to element
        @type force_set:    bool
        @param force_set:   Use javascript if True, webdriver if False
        """

        if val is None:
            val = ""

        self.click(force_click=True if force_set else False)
        self.clear()
        self.send_keys(val)
        actual = self.get_attribute('value')
        if val != actual:
            if force_set:
                js_executor = self.driver_wrapper.js_executor

                def force_set_element():
                    """
                    Wrapper to force_set element via javascript if needed
                    """
                    js_executor.execute_template('setValueTemplate', {'value': val}, self.element)
                    return True
                self.execute_and_handle_webelement_exceptions(force_set_element, 'set element by javascript')
            else:
                self.driver_wrapper.assertion.fail(
                    'Setting text field failed because final text did not match input value: "{}" != "{}"'.format(
                        actual,
                        val
                    )
                )
        return self