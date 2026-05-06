def archive_insert_data(self, data_dump):
        '''
        :param data: Archive table data
        :type data: list[archive]
        :raises: IOError
        '''
        with self.session as session:
            try:
                data = [self.tables.archive(**entry) for entry in data_dump]

                session.add_all(data)
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                print_exc()
                raise IOError(exc)