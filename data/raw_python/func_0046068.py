def set_decimal_value(self, value=None):
        """stub"""
        if value is None:
            raise NullArgument()
        if self.get_decimal_value_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_decimal(
                value,
                self.get_decimal_value_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['decimalValue'] = float(value)