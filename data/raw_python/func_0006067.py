def find_once(self, locator):
        """
        Find wrapper to run a single find

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         locator used in search
        @type find_all:         bool
        @param find_all:        should I find all elements, or just one?

        @rtype:                 WebElementWrapper or list[WebElementWrapper]
        @return:                Either a single WebElementWrapper, or a list of WebElementWrappers
        """
        params = []
        params.append(self.driver_wrapper.find_attempts)
        params.append(self.driver_wrapper.implicit_wait)

        self.driver_wrapper.find_attempts = 1
        self.driver_wrapper.implicit_wait = 0

        result = self.driver_wrapper._find_immediately(locator, self.element)

        # restore the original params
        self.driver_wrapper.implicit_wait = params.pop()
        self.driver_wrapper.find_attempts = params.pop()

        return result