def css(self, css_path, dom=None):
        """css find function abbreviation"""
        if dom is None:
            dom = self.browser
        return expect(dom.find_by_css, args=[css_path])