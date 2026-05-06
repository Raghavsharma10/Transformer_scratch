def find(self, locator, find_all=False, search_object=None, exclude_invisible=None, *args, **kwargs):
        """
        Find wrapper, invokes webDriverWrapper find with the current element as the search object

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used in search
        @type find_all:         bool
        @param find_all:        should I find all elements, or just one?
        @type search_object:    WebElementWrapper
        @param search_object:   Used to override the starting point of the driver search

        @rtype:                 WebElementWrapper or list[WebElementWrapper]
        @return:                Either a single WebElementWrapper, or a list of WebElementWrappers
        """
        search_object = self.element if search_object is None else search_object
        return self.driver_wrapper.find(
            locator,
            find_all,
            search_object=search_object,
            exclude_invisible=exclude_invisible
        )