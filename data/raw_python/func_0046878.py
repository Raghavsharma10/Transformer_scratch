def set_n_of_m(self, value=None):
        """stub"""
        if value is None:
            raise NullArgument()
        if isinstance(value, bool):
            # because True / False are also int types...
            raise InvalidArgument('value must be integer')
        if value is not None and not isinstance(value, int):
            raise InvalidArgument('value must be integer')
        if self.get_n_of_m_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_integer(value,
                                                          self.get_n_of_m_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['nOfM'] = value