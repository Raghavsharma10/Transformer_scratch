def add_target(self, target_text, name='', drop_behavior_type=None):
        """stub"""
        if not isinstance(target_text, DisplayText):
            raise InvalidArgument('target_text is not a DisplayText object')
        if not isinstance(name, DisplayText):
            # if default ''
            name = self._str_display_text(name)
        target = {
            'id': str(ObjectId()),
            'texts': [self._dict_display_text(target_text)],
            'names': [self._dict_display_text(name)],
            'dropBehaviorType': drop_behavior_type
        }
        self.my_osid_object_form._my_map['targets'].append(target)
        return target