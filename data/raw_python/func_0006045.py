def delete_content(self, max_chars=100):
        """
        Deletes content in the input field by repeatedly typing HOME, then DELETE

        @rtype:     WebElementWrapper
        @return:    Returns itself
        """
        def delete_content_element():
            chars_deleted = 0
            while len(self.get_attribute('value')) > 0 and chars_deleted < max_chars:
                self.click()
                self.send_keys(Keys.HOME)
                self.send_keys(Keys.DELETE)
                chars_deleted += 1

        self.execute_and_handle_webelement_exceptions(delete_content_element, 'delete input contents')
        return self