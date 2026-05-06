def send_special_keys(self, value):
        """
        Send special keys such as <enter> or <delete>

        @rtype:     WebElementWrapper
        @return:    Self
        """
        def send_keys_element():
            """
            Wrapper to send keys
            """
            return self.element.send_keys(value)
        self.execute_and_handle_webelement_exceptions(send_keys_element, 'send keys')
        return self