def set_min_string_length(self, length=None):
        """stub"""
        if self.get_min_string_length_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_cardinal(
                length,
                self.get_min_string_length_metadata()):
            raise InvalidArgument()
        if self.my_osid_object_form.max_string_length is not None and \
                length > self.my_osid_object_form.max_string_length - 1:
            raise InvalidArgument()
        self.my_osid_object_form._my_map['minStringLength'] = length
        self._min_string_length = length