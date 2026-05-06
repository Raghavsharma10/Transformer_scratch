def new_table(self, name, add_id=True, **kwargs):
        '''Add a table to the schema, or update it it already exists.

        If updating, will only update data.
        '''
        from . import Table
        from .exc import NotFoundError

        try:
            table = self.table(name)
            extant = True
        except NotFoundError:

            extant = False

            if 'sequence_id' not in kwargs:
                kwargs['sequence_id'] = self._database.next_sequence_id(Dataset, self.vid, Table)

            table = Table(name=name, d_vid=self.vid, **kwargs)

            table.update_id()

        # Update possibly extant data
        table.data = dict(
            (list(table.data.items()) if table.data else []) + list(kwargs.get('data', {}).items()))

        for key, value in list(kwargs.items()):

            if not key:
                continue
            if key[0] != '_' and key not in ['vid', 'id', 'id_', 'd_id', 'name', 'sequence_id', 'table', 'column', 'data']:
                setattr(table, key, value)

        if add_id:
            table.add_id_column()

        if not extant:
            self.tables.append(table)

        return table