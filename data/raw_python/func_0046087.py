def add_attempts(self, attempts):
        """stub"""
        if attempts is None:
            raise NullArgument('attempts cannot be None')
        if not self.my_osid_object_form._is_valid_integer(
                attempts, self.get_attempts_metadata()):
            raise InvalidArgument('attempts')
        self.my_osid_object_form._my_map['attempts'] = attempts