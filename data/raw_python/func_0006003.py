def find_all(self, locator, search_object=None, force_find=False):
        '''
        Find all elements matching locator

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         Locator object describing

        @rtype:                 list[WebElementWrapper]
        @return:                list of WebElementWrappers
        '''
        return self.find(locator=locator, find_all=True, search_object=search_object, force_find=force_find)