def _load_data(self, data, from_db=False):
        """
        Stores the data at self._data, actual object creation done at _generate_instances()

        Args:
            data (list): List of dicts.
            from_db (bool): Default False. Is this data coming from DB or not.
        """
        self._data = data[:]
        self.setattrs(
            values=[],
            node_stack=[],
            node_dict={},
        )
        self._from_db = from_db