def clean(self):
        """
        Goes through the txt input and cleans up any problematic HTML.
        """
        # Calls handle_starttag, handle_endtag, and handle_data
        self.feed()

        # Clean up any parent tags left open
        if self.current_parent_element['tag'] != '':
            self.cleaned_html += '</{}>'.format(self.current_parent_element['tag'])

        # Remove empty <p> added after lists
        self.cleaned_html = re.sub(r'(</[u|o]l>)<p></p>', r'\g<1>', self.cleaned_html)

        self._remove_pre_formatting()

        return self.cleaned_html