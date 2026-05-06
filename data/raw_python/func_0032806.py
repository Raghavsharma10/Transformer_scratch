def group_data(self):
        """All data about the current group (get-only).

        :getter: Returns all data about the current group
        :type: GroupData
        """
        return GroupData(self.group_num,
                         self.group_name,
                         self.group_symbol,
                         self.group_variant)