def is_present(self, locator, search_object=None):
        """
        Determines whether an element is present on the page, retrying once if unable to locate

        @type locator:                  webdriverwrapper.support.locator.Locator
        @param locator:                 the locator or css string used to query the element
        @type search_object:            webdriverwrapper.WebElementWrapper
        @param search_object:           Optional WebElement to start search with.
                                        If null, search will be on self.driver
        """
        all_elements = self._find_immediately(locator, search_object=search_object)

        if all_elements is not None and len(all_elements) > 0:
            return True
        else:
            return False