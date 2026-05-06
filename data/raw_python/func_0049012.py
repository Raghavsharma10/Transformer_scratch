def clear_start_timestamp(self):
        """stub"""
        if (self.get_start_timestamp_metadata().is_read_only() or
                self.get_start_timestamp_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['startTimestamp'] = \
            self.get_start_timestamp_metadata().get_default_integer_values()