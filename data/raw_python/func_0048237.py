def add_choice(self, choice, name='', identifier=None):
        """stub"""
        if not isinstance(choice, DisplayText):
            raise InvalidArgument('choice is not a displayText object')
        if identifier is None:
            identifier = str(ObjectId())
        current_identifiers = [c['id'] for c in self.my_osid_object_form._my_map['choices']]
        if identifier not in current_identifiers:
            choice = {
                'id': identifier,
                'texts': [self._dict_display_text(choice)],
                'name': name
            }
            self.my_osid_object_form._my_map['choices'].append(choice)
        else:
            for current_choice in self.my_osid_object_form._my_map['choices']:
                if current_choice['id'] == identifier:
                    self.add_or_replace_value('texts', choice, dictionary=current_choice)
                    choice = current_choice
        return choice