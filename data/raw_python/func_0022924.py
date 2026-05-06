def search_name(self, name, dom=None):
        """name find function abbreviation"""
        if dom is None:
            dom = self.browser
        return expect(dom.find_by_name, args=[name])