def add_showanswer(self, showanswer):
        """stub"""
        if showanswer is None:
            raise NullArgument('showanswer cannot be None')
        if not self.my_osid_object_form._is_valid_string(
                showanswer, self.get_showanswer_metadata()):
            raise InvalidArgument('showanswer')
        self.my_osid_object_form._my_map['showanswer'] = showanswer