def highlight(self):
        """
        Draws a dotted red box around the wrapped element using javascript

        @rtype:     WebElementWrapper
        @return:    Self
        """
        js_executor = self.driver_wrapper.js_executor
        def highlight_element():
            """
            Wrapper to highlight elements
            """
            location = self.element.location
            size = self.element.size
            js_executor.execute_template('elementHighlighterTemplate', {
                'x': str(location['x']),
                'y': str(location['y']),
                'width': str(size['width']),
                'height': str(size['height'])})
            return True
        self.execute_and_handle_webelement_exceptions(highlight_element, 'highlight')
        return self