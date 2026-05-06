def xpath(self, xpath, dom=None):
        """xpath find function abbreviation"""
        if dom is None:
            dom = self.browser
        return expect(dom.find_by_xpath, args=[xpath])