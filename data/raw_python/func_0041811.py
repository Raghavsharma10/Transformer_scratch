def bulk_archive_rows(cls, rows, session, user_id=None, chunk_size=1000, commit=True):
        """
        Bulk archives data previously written to DB.

        :param rows: iterable of previously saved model instances to archive
        :param session: DB session to use for inserts
        :param user_id: ID of user responsible for row modifications
        :return:
        """
        dialect = utils.get_dialect(session)
        to_insert_dicts = []
        for row in rows:
            row_dict = cls.build_row_dict(row, user_id=user_id, dialect=dialect)
            to_insert_dicts.append(row_dict)
            if len(to_insert_dicts) < chunk_size:
                continue

            # Insert a batch of rows
            session.execute(insert(cls).values(to_insert_dicts))
            to_insert_dicts = []

        # Insert final batch of rows (if any)
        if to_insert_dicts:
            session.execute(insert(cls).values(to_insert_dicts))
        if commit:
            session.commit()