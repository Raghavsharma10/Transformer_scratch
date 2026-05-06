def changed_fields(self, from_db=False):
        """
        Args:
            from_db (bool): Check changes against actual db data
        Returns:
            list: List of fields names which their values changed.
        """
        if self.exist:
            current_dict = self.clean_value()
            # `from_db` attr is set False as default, when a `ListNode` is
            # initialized just after above `clean_value` is called. `from_db` flags
            # in 'list node sets' makes differences between clean_data and object._data.

            db_data = self._initial_data
            if from_db:
                # Thus, after clean_value, object's data is taken from db again.
                db_data = self.objects.data().get(self.key)[0]

            set_current, set_past = set(current_dict.keys()), set(db_data.keys())
            intersect = set_current.intersection(set_past)
            return set(o for o in intersect if db_data[o] != current_dict[o])