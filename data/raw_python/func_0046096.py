def clear_time_value(self):
        """stub"""
        if (self.get_time_value_metadata().is_read_only() or
                self.get_time_value_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['timeValue'] = \
            dict(self.get_time_value_metadata().get_default_duration_values()[0])