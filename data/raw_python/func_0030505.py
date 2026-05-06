def command_drop_tables(self, meta_name=None):
        '''
        Drops all tables without dropping a database::

            ./manage.py sqla:drop_tables [meta_name]
        '''
        answer = six.moves.input(u'All data will lost. Are you sure? [y/N] ')

        if answer.strip().lower()!='y':
            sys.exit('Interrupted')

        def _drop_metadata_tables(metadata):
            table = next(six.itervalues(metadata.tables), None)
            if table is None:
                print('Failed to find engine')
            else:
                engine = self.session.get_bind(clause=table)
                drop_everything(engine)
                print('Done')

        if isinstance(self.metadata, MetaData):
            print('Droping tables... ', end='')
            _drop_metadata_tables(self.metadata)
        else:
            for current_meta_name, metadata in self.metadata.items():
                if meta_name not in (current_meta_name, None):
                    continue
                print('Droping tables for {}... '.format(current_meta_name),
                      end='')
                _drop_metadata_tables(metadata)