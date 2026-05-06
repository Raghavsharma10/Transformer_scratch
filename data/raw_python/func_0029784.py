def add_column(self, name, update_existing=False, **kwargs):
        """
        Add a column to the table, or update an existing one.
        :param name: Name of the new or existing column.
        :param update_existing: If True, alter existing column values. Defaults to False
        :param kwargs: Other arguments for the the Column() constructor
        :return: a Column object
        """
        from ..identity import ColumnNumber

        try:
            c = self.column(name)
            extant = True

            if not update_existing:
                return c

        except NotFoundError:

            sequence_id = len(self.columns) + 1

            assert sequence_id

            c = Column(t_vid=self.vid,
                       sequence_id=sequence_id,
                       vid=str(ColumnNumber(ObjectNumber.parse(self.vid), sequence_id)),
                       name=name,
                       datatype='str')
            extant = False

        # Update possibly existing data
        c.data = dict((list(c.data.items()) if c.data else []) + list(kwargs.get('data', {}).items()))

        for key, value in list(kwargs.items()):

            if key[0] != '_' and key not in ['t_vid', 'name',  'sequence_id', 'data']:

                # Don't update the type if the user has specfied a custom type
                if key == 'datatype' and not c.type_is_builtin():
                    continue

                # Don't change a datatype if the value is set and the new value is unknown
                if key == 'datatype' and value == 'unknown' and c.datatype:
                    continue

                # Don't change a datatype if the value is set and the new value is unknown
                if key == 'description' and not value:
                    continue

                try:
                    setattr(c, key, value)
                except AttributeError:
                    raise AttributeError("Column record has no attribute {}".format(key))

            if key == 'is_primary_key' and isinstance(value, str) and len(value) == 0:
                value = False
                setattr(c, key, value)

        # If the id column has a description and the table does not, add it to
        # the table.
        if c.name == 'id' and c.is_primary_key and not self.description:
            self.description = c.description

        if not extant:
            self.columns.append(c)

        return c