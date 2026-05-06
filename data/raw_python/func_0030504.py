def command_create_tables(self, meta_name=None, verbose=False):
        '''
        Create tables according sqlalchemy data model.

        Is not a complex migration tool like alembic, just creates tables that
        does not exist::

            ./manage.py sqla:create_tables [--verbose] [meta_name]
        '''

        def _create_metadata_tables(metadata):
            for table in metadata.sorted_tables:
                if verbose:
                    print(self._schema(table))
                else:
                    print('  '+table.name)
                engine = self.session.get_bind(clause=table)
                metadata.create_all(bind=engine, tables=[table])

        if isinstance(self.metadata, MetaData):
            print('Creating tables...')
            _create_metadata_tables(self.metadata)
        else:
            for current_meta_name, metadata in self.metadata.items():
                if meta_name not in (current_meta_name, None):
                    continue
                print('Creating tables for {}...'.format(current_meta_name))
                _create_metadata_tables(metadata)