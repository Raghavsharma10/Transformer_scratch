def get_attribute(self, name):
        """
        Retrieves specified attribute from WebElement

        @type name:     str
        @param name:    Attribute to retrieve

        @rtype:         str
        @return:        String representation of the attribute
        """
        def get_attribute_element():
            """
            Wrapper to retrieve element
            """
            return self.element.get_attribute(name)
        return self.execute_and_handle_webelement_exceptions(get_attribute_element, 'get attribute "' + str(name) + '"')