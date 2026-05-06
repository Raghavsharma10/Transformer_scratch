def _find_immediately(self, locator, search_object=None):
        '''
        Attempts to immediately find elements on the page without waiting

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         Locator object describing
        @type search_object:    webdriverwrapper.WebElementWrapper
        @param search_object:   Optional WebElement to start search with.  If null, search will be on self.driver


        @return:                Single WebElemetnWrapper if find_all is False,
                                list of WebElementWrappers if find_all is True
        '''
        search_object = self.driver if search_object is None else search_object
        elements = self.locator_handler.find_by_locator(search_object, locator, True)
        return [WebElementWrapper.WebElementWrapper(self, locator, element) for element in elements]