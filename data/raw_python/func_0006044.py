def clear(self):
        """
        Clears the field represented by this element

        @rtype:     WebElementWrapper
        @return:    Returns itself
        """
        def clear_element():
            """
            Wrapper to clear element
            """
            return self.element.clear()
        self.execute_and_handle_webelement_exceptions(clear_element, 'clear')
        return self