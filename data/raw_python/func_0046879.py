def clear_n_of_m(self):
        """stub"""
        if (self.get_n_of_m_metadata().is_read_only() or
                self.get_n_of_m_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['nOfM'] = \
            int(self._n_of_m_metadata['default_object_values'][0])