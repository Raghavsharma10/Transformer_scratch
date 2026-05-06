def is_present_no_wait(self, locator):
        """
        Determines whether an element is present on the page with no wait

        @type locator:  webdriverwrapper.support.locator.Locator
        @param locator: the locator or css string used to query the element
        """

        # first attempt to locate the element

        def execute():
            '''
            Generic function to execute wait
            '''
            return True if len(self.locator_handler.find_by_locator(self.driver, locator, True)) < 0 else False

        return self.execute_and_handle_webdriver_exceptions(
            execute, timeout=0, locator=locator, failure_message='Error running webdriver.find_all.')