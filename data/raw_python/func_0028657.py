def _bind_set_table_metadata(self, key,  multiline=False):
        """Returns parsing function which will parse data as text, and add it to the table metatadata dictionary
        with the provided key

        :param key: dictionary key under which parsed data will be added to table.metadata
        :type key: str
        :param multiline: if True this attribute will be treated as multiline
        :type multiline: bool
        :return: function with bound key and multiline attributes
        :rtype: Function
        """
        def set_table_metadata(self, data):
            if multiline:
                data = self._read_multiline(data)
            if key == 'location' and data:
                data = 'Data from ' + data
            self.current_table.metadata[key] = data.strip()

        # method must be bound, so we use __get__
        return set_table_metadata.__get__(self)