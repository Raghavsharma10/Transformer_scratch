def set_time_value(self, value=None):
        """stub"""
        if value is None:
            raise NullArgument()
        if self.get_time_value_metadata().is_read_only():
            raise NoAccess()
        if self.my_osid_object_form._is_valid_duration(
                value,
                self.get_time_value_metadata()):
            # http://stackoverflow.com/questions/775049/python-time-seconds-to-hms
            time = self._convert_duration_to_hhmmss(value)
        elif utilities.is_string(value):
            # assume something like hh:mm:ss, convert to dict
            time = self._convert_string_to_hhmmss(value)
        else:
            raise InvalidArgument('value must be a string or duration')
        self.my_osid_object_form._my_map['timeValue'] = {
            'hours': time['hours'],
            'minutes': time['minutes'],
            'seconds': time['seconds']
        }