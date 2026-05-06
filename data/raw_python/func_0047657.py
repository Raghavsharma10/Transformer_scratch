def set_difficulty(self, difficulty):
        """stub"""
        if not is_string(difficulty):
            raise InvalidArgument('difficulty value must be a string')
        if difficulty.lower() not in ['low', 'medium', 'hard']:
            raise InvalidArgument('difficulty value must be low, medium, or hard')
        self.my_osid_object_form._my_map['texts']['difficulty']['text'] = difficulty