def click(self, force_click=False):
        """
        Clicks the element

        @type force_click:  bool
        @param force_click: force a click on the element using javascript, skipping webdriver

        @rtype:             WebElementWrapper
        @return:            Returns itself
        """
        js_executor = self.driver_wrapper.js_executor

        def click_element():
            """
            Wrapper to call click
            """
            return self.element.click()

        def force_click_element():
            """
            Javascript wrapper to force_click the element
            """
            js_executor.execute_template('clickElementTemplate', {}, self.element)
            return True

        if force_click:
            self.execute_and_handle_webelement_exceptions(force_click_element, 'click element by javascript')
        else:
            self.execute_and_handle_webelement_exceptions(click_element, 'click')

        return self