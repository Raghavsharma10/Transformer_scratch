def set_end_timestamp(self, end_timestamp=None):
        """stub"""
        if end_timestamp is None:
            raise NullArgument()
        if self.get_end_timestamp_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_integer(
                end_timestamp,
                self.get_end_timestamp_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['endTimestamp'] = end_timestamp