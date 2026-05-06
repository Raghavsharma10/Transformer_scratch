def _close_list(self):
        """
        Add an close list tag corresponding to the currently open
        list found in current_parent_element.
        """
        list_type = self.current_parent_element['attrs']['class']
        tag = LIST_TYPES[list_type]

        html = '</{t}>'.format(
            t=tag
        )
        self.cleaned_html += html
        self.current_parent_element['tag'] = ''
        self.current_parent_element['attrs'] = {}