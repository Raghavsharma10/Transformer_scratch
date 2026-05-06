def set_max_attempts(self, value):
        """stub"""
        if value is None:
            raise InvalidArgument('value must be an integer')
        if value is not None and not isinstance(value, int):
            raise InvalidArgument('value is not an integer')
        if not self.my_osid_object_form._is_valid_integer(value,
                                                          self.get_max_attempts_metadata()):
            raise InvalidArgument('value must be an integer')
        self.my_osid_object_form._my_map['maxAttempts'] = value