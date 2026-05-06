def set_start_timestamp(self, start_timestamp=None):
        """stub"""
        if start_timestamp is None:
            raise NullArgument()
        if self.get_start_timestamp_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_integer(
                start_timestamp,
                self.get_start_timestamp_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['startTimestamp'] = start_timestamp