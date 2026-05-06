def command_schema(self, name=None):
        '''
        Prints current database schema (according sqlalchemy database model)::

            ./manage.py sqla:schema [name]
        '''
        meta_name = table_name = None
        if name:
            if isinstance(self.metadata, MetaData):
                table_name = name
            elif '.' in name:
                meta_name, table_name = name.split('.', 1)
            else:
                meta_name = name

        def _print_metadata_schema(metadata):
            if table_name is None:
                for table in metadata.sorted_tables:
                    print(self._schema(table))
            else:
                try:
                    table = metadata.tables[table_name]
                except KeyError:
                    sys.exit('Table {} is not found'.format(name))
                print(self._schema(table))

        if isinstance(self.metadata, MetaData):
            _print_metadata_schema(self.metadata)
        else:
            for current_meta_name, metadata in self.metadata.items():
                if meta_name not in (current_meta_name, None):
                    continue
                _print_metadata_schema(metadata)