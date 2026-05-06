def _create_spec_config(self, table_name, spec_documents):
        '''
        Dynamo implementation of spec config creation

        Called by `create_archive_table()` in
        :py:class:`manager.BaseDataManager` Simply adds two rows to the spec
        table

        Parameters
        ----------

        table_name :

            base table name (not including .spec suffix)

        spec_documents : list

            list of dictionary documents defining the manager spec


        '''

        _spec_table = self._resource.Table(table_name + '.spec')

        for doc in spec_documents:
            _spec_table.put_item(Item=doc)