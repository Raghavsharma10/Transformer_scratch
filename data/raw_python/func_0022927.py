def elXpath(self, xpath, dom=None):
        """check if element is present by css"""
        if dom is None:
            dom = self.browser
        return expect(dom.is_element_present_by_xpath, args=[xpath])