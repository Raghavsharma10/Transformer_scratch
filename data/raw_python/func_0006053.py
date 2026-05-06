def value_of_css_property(self, property_name):
        """
        Get value of CSS property for element

        @rtype:     str
        @return:    value of CSS property
        """
        def value_of_css_property_element():
            """
            Wrapper to get css property
            """
            return self.element.value_of_css_property(property_name)
        return self.execute_and_handle_webelement_exceptions(value_of_css_property_element, 'get css property "' +
                                                                                           str(property_name) + '"')