def hover(self):
        """
        Hovers the element
        """
        def do_hover():
            """
            Perform hover
            """
            ActionChains(self.driver_wrapper.driver).move_to_element(self.element).perform()
        return self.execute_and_handle_webelement_exceptions(do_hover, 'hover')