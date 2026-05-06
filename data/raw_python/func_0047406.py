def clear_feedbacks(self):
        """stub"""
        if self.get_feedbacks_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['feedbacks'] = \
            self._feedbacks_metadata['default_object_values'][0]