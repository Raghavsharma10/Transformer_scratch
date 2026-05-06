def _set_table(self, data):
        """Set current parsing state to 'table',
        create new table object and add it to tables collection
        """
        self.set_state('table')
        self.current_table = HEPTable(index=len(self.tables) + 1)
        self.tables.append(self.current_table)
        self.data.append(self.current_table.metadata)