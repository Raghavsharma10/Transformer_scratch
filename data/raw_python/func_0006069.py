def is_present(self, locator):
        """
        Tests to see if an element is present

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used in search

        @rtype:                 bool
        @return:                True if present, False if not present
        """
        return self.driver_wrapper.is_present(locator, search_object=self.element)