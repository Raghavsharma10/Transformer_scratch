def to_html(self, css_class=''):
        """
        Returns the parameter as a dt/dd pair.
        """
        if self.name and self.type:
            header_text = '%s (%s)' % (self.name, self.type)
        elif self.type:
            header_text = self.type
        else:
            header_text = self.name
        return '<dt>%s</dt><dd>%s</dd>' % (header_text, self.doc)