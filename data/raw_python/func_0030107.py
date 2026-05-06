def table(self, ref):
        """ Finds table by ref and returns it.

        Args:
            ref (str): id, vid (versioned id) or name of the table

        Raises:
            NotFoundError: if table with given ref not found.

        Returns:
            orm.Table

        """

        try:
            obj_number = ObjectNumber.parse(ref)
            ds_obj_number = obj_number.as_dataset

            dataset = self._db.dataset(ds_obj_number)  # Could do it in on SQL query, but this is easier.
            table = dataset.table(ref)

        except NotObjectNumberError:
            q = self.database.session.query(Table)\
                .filter(Table.name == str(ref))\
                .order_by(Table.vid.desc())

            table = q.first()

        if not table:
            raise NotFoundError("No table for ref: '{}'".format(ref))
        return table