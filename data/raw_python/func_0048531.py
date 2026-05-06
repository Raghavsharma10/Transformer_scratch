def add_droppable(self, droppable_text, name='', reuse=1, drop_behavior_type=None):
        """stub"""
        if not isinstance(droppable_text, DisplayText):
            raise InvalidArgument('droppable_text is not a DisplayText object')
        if not isinstance(reuse, int):
            raise InvalidArgument('reuse must be an integer')
        if reuse < 0:
            raise InvalidArgument('reuse must be >= 0')
        if not isinstance(name, DisplayText):
            # if default ''
            name = self._str_display_text(name)
        droppable = {
            'id': str(ObjectId()),
            'texts': [self._dict_display_text(droppable_text)],
            'names': [self._dict_display_text(name)],
            'reuse': reuse,
            'dropBehaviorType': drop_behavior_type
        }
        self.my_osid_object_form._my_map['droppables'].append(droppable)
        return droppable