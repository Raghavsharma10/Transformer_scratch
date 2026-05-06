def groups_names(self):
        """Names of all groups (get-only).

        :getter: Returns names of all groups
        :type: list of str
        """
        return _ListProxy(self._get_group_name_by_num(i) for i in range(self.groups_count))