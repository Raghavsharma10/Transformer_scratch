def groups_data(self):
        """All data about all groups (get-only).

        :getter: Returns all data about all groups
        :type: list of GroupData
        """
        return _ListProxy(GroupData(num, name, symbol, variant)
                          for (num, name, symbol, variant)
                          in zip(range(self.groups_count),
                                 self.groups_names,
                                 self.groups_symbols,
                                 self.groups_variants))