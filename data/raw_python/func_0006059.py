def set_attribute(self, name, value):
        """Sets the attribute of the element to a specified value

        @type name:     str
        @param name:    the name of the attribute
        @type value:    str
        @param value:   the attribute of the value
        """
        js_executor = self.driver_wrapper.js_executor
        def set_attribute_element():
            """
            Wrapper to set attribute
            """
            js_executor.execute_template('setAttributeTemplate', {
                'attribute_name': str(name),
                'attribute_value': str(value)}, self.element)
            return True
        self.execute_and_handle_webelement_exceptions(set_attribute_element,
                                                      'set attribute "' + str(name) + '" to "' + str(value) + '"')
        return self