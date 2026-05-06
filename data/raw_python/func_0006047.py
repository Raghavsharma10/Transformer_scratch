def get_value(self):
        """Gets the value of a select or input element

        @rtype: str
        @return: The value of the element
        @raise: ValueError if element is not of type input or select, or has multiple selected options
        """
        def get_element_value():
            if self.tag_name() == 'input':
                return self.get_attribute('value')
            elif self.tag_name() == 'select':
                selected_options = self.element.all_selected_options
                if len(selected_options) > 1:
                    raise ValueError(
                        'Select {} has multiple selected options, only one selected '
                        'option is valid for this method'.format(self)
                    )
                return selected_options[0].get_attribute('value')
            else:
                raise ValueError('Can not get the value of elements or type "{}"'.format(self.tag_name()))

        return self.execute_and_handle_webelement_exceptions(get_element_value, name_of_action='get value')