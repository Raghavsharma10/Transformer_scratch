def text(self, force_get=False):
        """
        Get the text of the element

        @rtype:     str
        @return:    Text of the element
        """
        def text_element():
            """
            Wrapper to get text of element
            """
            return self.element.text

        def force_text_element():
            """Get text by javascript"""
            return self.driver_wrapper.js_executor.execute_template_and_return_result(
                'getElementText.js', {}, self.element
            )

        if force_get:
            return self.execute_and_handle_webelement_exceptions(force_text_element, 'get text by javascript')
        else:
            return self.execute_and_handle_webelement_exceptions(text_element, 'get text')