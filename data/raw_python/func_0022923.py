def css1(self, css_path, dom=None):
        """return the first value of self.css"""
        if dom is None:
            dom = self.browser

        def _css1(path, domm):
            """virtual local func"""
            return self.css(path, domm)[0]

        return expect(_css1, args=[css_path, dom])