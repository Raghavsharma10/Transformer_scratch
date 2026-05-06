def insert(self, row):
        """
        Add a row (type: dict) by inserting it into the table.
        Columns must exist.
        ::
            data = dict(title='I am a banana!')
            table.insert(data)
        Returns the inserted row's primary key.
        """
        self._check_dropped()
        res = self.engine.execute(self.table.insert(row))
        if len(res.inserted_primary_key) > 0:
            return res.inserted_primary_key[0]