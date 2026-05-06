def submit(self):
        """
        Submit a webe element

        @rtype:     WebElementWrapper
        @return:    Self
        """
        def submit_element():
            """
            Wrapper to submit element
            """
            return self.element.submit()
        self.execute_and_handle_webelement_exceptions(submit_element, 'send keys')
        return self