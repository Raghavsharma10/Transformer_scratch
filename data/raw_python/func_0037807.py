def set_primary_key_auto(self, table):
        """
        Analysis a table and set a primary key.

        Determine primary key by identifying a column with unique values
        or creating a new column.

        :param table: Table to alter
        :return: Primary Key column
        """
        # Confirm no primary key exists
        pk = self.get_primary_key(table)
        if not pk:
            # Determine if there is a unique column that can become the PK
            unique_col = self.get_unique_column(table)

            # Set primary key
            if unique_col:
                self.set_primary_key(table, unique_col)

            # Create unique 'ID' column
            else:
                unique_col = self.add_column(table, primary_key=True)
            return unique_col
        else:
            return pk