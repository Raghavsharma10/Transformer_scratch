def handle_endtag(self, tag):
        """
        Called by HTMLParser.feed when an end tag is found.
        """
        if tag in PARENT_ELEMENTS:
            self.current_parent_element['tag'] = ''
            self.current_parent_element['attrs'] = ''

        if tag == 'li':
            self.parsing_li = True
        if tag != 'br':
            self.cleaned_html += '</{}>'.format(tag)