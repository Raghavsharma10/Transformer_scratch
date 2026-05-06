def add_table_item(self, table):
        """
        Adds a Table to the publish group.
        """
        if not table.is_draft_version:
            raise ValueError("Table isn't a draft version")

        self.items.append(table.latest_version)