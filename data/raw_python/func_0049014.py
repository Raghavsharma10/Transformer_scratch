def clear_end_timestamp(self):
        """stub"""
        if (self.get_end_timestamp_metadata().is_read_only() or
                self.get_end_timestamp_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['endTimestamp'] = \
            self.get_end_timestamp_metadata().get_default_integer_values()