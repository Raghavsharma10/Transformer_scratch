def set_published(self, value=None):
        """stub"""
        if value is None:
            raise NullArgument()
        if self.get_published_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_boolean(value):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['published'] = value