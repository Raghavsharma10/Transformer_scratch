def archive_query_interval(self, _from, to):
        '''
        :param _from: Start of interval (int) (inclusive)
        :param to: End of interval (int) (exclusive)
        :raises: IOError
        '''
        with self.session as session:
            table = self.tables.archive

            try:
                results = session.query(table)\
                    .filter(table.dateTime >= _from)\
                    .filter(table.dateTime < to)\
                    .all()

                return [self.archive_schema.dump(entry).data for entry in results]
            except SQLAlchemyError as exc:
                session.rollback()
                print_exc()
                raise IOError(exc)