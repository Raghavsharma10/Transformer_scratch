def add_choice(self, text, name='', identifier=None):
        """stub"""
        if not utilities.is_string(text):
            raise InvalidArgument('text is not a string')
        choice_display_text = self._choice_text_metadata['default_string_values'][0]
        choice_display_text['text'] = text
        if identifier is None:
            identifier = str(ObjectId())
        choice = {
            'id': identifier,
            'text': text,
            'name': name
        }
        self.my_osid_object_form._my_map['choices'].append(choice)
        return choice