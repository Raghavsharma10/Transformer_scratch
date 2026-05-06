def find_all(self, locator):
        """
        Find wrapper, finds all elements

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used in search

        @rtype:                 list
        @return:                A list of WebElementWrappers
        """
        return self.driver_wrapper.find(locator, True, self.element)