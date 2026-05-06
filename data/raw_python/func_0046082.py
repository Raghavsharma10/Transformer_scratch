def add_decimal_value(self, value, label=None):
        """stub"""
        if label is None:
            label = self._label_metadata['default_string_values'][0]
        else:
            if not self.my_osid_object_form._is_valid_string(
                    label, self.get_label_metadata()) or '.' in label:
                raise InvalidArgument('label')
        if value is None:
            raise NullArgument('value cannot be None')
        if not self.my_osid_object_form._is_valid_decimal(
                value, self.get_decimal_value_metadata()):
            raise InvalidArgument('value')
        self.my_osid_object_form._my_map['decimalValues'][label] = value