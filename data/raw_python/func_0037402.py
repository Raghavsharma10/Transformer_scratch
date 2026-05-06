def _create_archive_table(self, table_name):
        '''
        Dynamo implementation of BaseDataManager create_archive_table

        waiter object is implemented to ensure table creation before moving on
        this will slow down table creation. However, since we are only creating
        table once this should no impact users.

        Parameters
        ----------
        table_name: str

        Returns
        -------
        None

        '''
        if table_name in self._get_table_names():
            raise KeyError('Table "{}" already exists'.format(table_name))

        try:
            table = self._resource.create_table(
                TableName=table_name,
                KeySchema=[{'AttributeName': '_id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[
                    {'AttributeName': '_id', 'AttributeType': 'S'}],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 123,
                    'WriteCapacityUnits': 123})

            table.meta.client.get_waiter('table_exists').wait(
                TableName=table_name)

        except ValueError:
            # Error handling for windows incompatability issue
            msg = 'Table creation failed'
            assert table_name in self._get_table_names(), msg