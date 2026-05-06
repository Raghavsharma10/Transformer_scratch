def clear_choice_texts(self, choice_id):
        """stub"""
        if self.get_choices_metadata().is_read_only():
            raise NoAccess()
        updated_choices = []
        for current_choice in self.my_osid_object_form._my_map['choices']:
            if current_choice['id'] != choice_id:
                updated_choices.append(current_choice)
            else:
                updated_choices.append({
                    'id': current_choice['id'],
                    'texts': [],
                    'name': current_choice['name']
                })
        self.my_osid_object_form._my_map['choices'] = updated_choices