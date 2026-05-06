def before_update(mapper, conn, target):
        """Set the Table ID based on the dataset number and the sequence number
        for the table."""

        target.name = Table.mangle_name(target.name)

        if isinstance(target, Column):
            raise TypeError('Got a column instead of a table')

        target.update_id(target.sequence_id, False)