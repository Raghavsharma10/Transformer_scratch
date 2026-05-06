def xsl_elements(self):
        """Find all "XSL" styled runs, normalize related paragraph and returns list of XslElements"""

        def append_xsl_elements(xsl_elements, r, xsl):
            if r is not None:
                r.xpath('.//w:t',  namespaces=self.namespaces)[0].text = xsl
                xe = XslElement(r, logger=self.logger)
                xsl_elements.append(xe)
            return None, ''

        if not getattr(self, '_xsl_elements', None):
            xsl_elements = []
            for p in self.root.xpath('.//w:p', namespaces=self.namespaces):
                xsl_r, xsl = None, ''
                for r in p:
                    # find first XSL run and add all XSL meta text
                    text = ''.join(t.text for t in r.xpath('.//w:t', namespaces=self.namespaces))
                    if r.xpath('.//w:rPr/w:rStyle[@w:val="%s"]' % self.style, namespaces=self.namespaces):
                        xsl += text
                        if xsl_r is None and text:
                            xsl_r = r
                        else:
                            r.getparent().remove(r)
                    elif text:
                        xsl_r, xsl = append_xsl_elements(xsl_elements, xsl_r, xsl)
                xsl_r, xsl = append_xsl_elements(xsl_elements, xsl_r, xsl)
            self._xsl_elements = xsl_elements

        return self._xsl_elements