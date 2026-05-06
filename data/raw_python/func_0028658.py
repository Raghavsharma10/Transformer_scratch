def _bind_parse_additional_data(self, key, multiline=False):
        """Returns parsing function which will parse data as text, and add it to the table additional data dictionary
        with the provided key

        :param key: dictionary key under which parsed data will be added to table.metadata
        :type key: str
        :param multiline: if True this attribute will be treated as multiline
        :type multiline: bool
        :return: function with bound key and multiline attributes
        :rtype: Function
        """

        def _set_additional_data_bound(self, data):
            """Concrete method for setting additional data
            :param self:
            :type self: OldHEPData
            """
            # if it's multiline, parse it
            if multiline:
                data = self._read_multiline(data)

            if key not in self.additional_data:
                self.additional_data[key] = []
            self.additional_data[key].append(data)

        # method must be bound, so we use __get__
        return _set_additional_data_bound.__get__(self)