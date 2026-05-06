def before_update(mapper, conn, target):
        """Set the column id number based on the table number and the sequence
        id for the column."""

        assert target.datatype or target.valuetype

        target.name = Column.mangle_name(target.name)

        Column.update_number(target)