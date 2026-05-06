def remove_style(self):
        """Remove all XSL run rStyle elements"""

        for n in self.root.xpath('.//w:rStyle[@w:val="%s"]' % self.style, namespaces=self.namespaces):
            n.getparent().remove(n)