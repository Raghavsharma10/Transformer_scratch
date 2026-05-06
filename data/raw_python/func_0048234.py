def get_choices(self):
        """stub"""
        # ideally would return a displayText object in text ... except for legacy
        # use cases like OEA, it expects a text string.
        choices = []
        # for current_choice in self.my_osid_object.object_map['choices']:
        for current_choice in self.my_osid_object._my_map['choices']:
            filtered_choice = {
                'id': current_choice['id'],
                'text': self.get_matching_language_value('texts',
                                                         dictionary=current_choice).text,
                'name': current_choice['name']
            }
            choices.append(filtered_choice)
        return choices