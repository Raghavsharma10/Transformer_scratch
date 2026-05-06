def parent(self):
        """
        Get the parent of the element

        @rtype:     WebElementWrapper
        @return:    Parent of webelementwrapper on which this was invoked
        """
        def parent_element():
            """
            Wrapper to retrieve parent element
            """
            return WebElementWrapper(self.driver_wrapper, self.locator, self.element.parent)
        return self.execute_and_handle_webelement_exceptions(parent_element, 'get parent')