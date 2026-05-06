def _open_list(self, list_type):
        """
        Add an open list tag corresponding to the specification in the
        parser's LIST_TYPES.
        """
        if list_type in LIST_TYPES.keys():
            tag = LIST_TYPES[list_type]
        else:
            raise Exception('CustomSlackdownHTMLParser:_open_list: Not a valid list type.')

        html = '<{t} class="list-container-{c}">'.format(
            t=tag,
            c=list_type
        )
        self.cleaned_html += html
        self.current_parent_element['tag'] = LIST_TYPES[list_type]
        self.current_parent_element['attrs'] = {'class': list_type}