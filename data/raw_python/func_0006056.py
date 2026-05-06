def parent_element(self):
        """
        Get the parent of the element

        @rtype:     WebElementWrapper
        @return:    Parent of webelementwrapper on which this was invoked
        """
        def parent_element():
            """
            Wrapper to get parent element
            """
            parent = self.driver_wrapper.execute_script('return arguments[0].parentNode;', self.element)
            wrapped_parent = WebElementWrapper(self.driver_wrapper, '', parent)
            return wrapped_parent

        return self.execute_and_handle_webelement_exceptions(parent_element, 'get parent element')