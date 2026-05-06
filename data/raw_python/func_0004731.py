def handle_starttag(self, tag, attrs):
        """
        Called by HTMLParser.feed when a start tag is found.
        """
        # Parse the tag attributes
        attrs_dict = dict(t for t in attrs)

        # If the tag is a predefined parent element
        if tag in PARENT_ELEMENTS:
            # If parser is parsing another parent element
            if self.current_parent_element['tag'] != '':
                # close the parent element
                self.cleaned_html += '</{}>'.format(self.current_parent_element['tag'])

            self.current_parent_element['tag'] = tag
            self.current_parent_element['attrs'] = {}

            self.cleaned_html += '<{}>'.format(tag)

        # If the tag is a list item
        elif tag == 'li':
            self.parsing_li = True

            # Parse the class name & subsequent type
            class_name = attrs_dict['class']
            list_type = class_name[10:]

            # Check if parsing a list
            if self.current_parent_element['tag'] == 'ul' or self.current_parent_element['tag'] == 'ol':
                cur_list_type = self.current_parent_element['attrs']['class']
                # Parsing a different list
                if cur_list_type != list_type:
                    # Close that list
                    self._close_list()

                    # Open new list
                    self._open_list(list_type)
            # Not parsing a list
            else:
                # if parsing some other parent
                if self.current_parent_element['tag'] != '':
                    self.cleaned_html += '</{}>'.format(self.current_parent_element['tag'])
                # Open new list
                self._open_list(list_type)

            self.cleaned_html += '<{}>'.format(tag)

        # If the tag is a line break
        elif tag == 'br':
            # If parsing a paragraph, close it
            if self.current_parent_element['tag'] == 'p':
                self.cleaned_html += '</p>'
                self.current_parent_element['tag'] = ''
                self.current_parent_element['attrs'] = {}
            # If parsing a list, close it
            elif self.current_parent_element['tag'] == 'ul' or self.current_parent_element['tag'] == 'ol':
                self._close_list()
            # If parsing any other parent element, keep it
            elif self.current_parent_element['tag'] in PARENT_ELEMENTS:
                self.cleaned_html += '<br />'
            # If not in any parent element, create an empty paragraph
            else:
                self.cleaned_html += '<p></p>'

        # If the tag is something else, like a <b> or <i> tag
        else:
            # If not parsing any parent element
            if self.current_parent_element['tag'] == '':
                self.cleaned_html += '<p>'
                self.current_parent_element['tag'] = 'p'
            self.cleaned_html += '<{}'.format(tag)

            for attr in sorted(attrs_dict.keys()):
                self.cleaned_html += ' {k}="{v}"'.format(
                    k=attr,
                    v=attrs_dict[attr]
                )

            self.cleaned_html += '>'