def elCss(self, css_path, dom=None):
        """check if element is present by css"""
        if dom is None:
            dom = self.browser
        return expect(dom.is_element_present_by_css, args=[css_path])