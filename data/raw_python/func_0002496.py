def _populate_stub(self, name, stub, table):
        """
        Populate the placeholders in the migration stub.

        :param name: The name of the migration
        :type name: str

        :param stub: The stub
        :type stub: str

        :param table: The table name
        :type table: str

        :rtype: str
        """
        stub = stub.replace('DummyClass', self._get_class_name(name))

        if table is not None:
            stub = stub.replace('dummy_table', table)

        return stub