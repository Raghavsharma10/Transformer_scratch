def handle_data(self, data):
        """
        Called by HTMLParser.feed when text is found.
        """
        if self.current_parent_element['tag'] == '':
            self.cleaned_html += '<p>'
            self.current_parent_element['tag'] = 'p'

        self.cleaned_html += data