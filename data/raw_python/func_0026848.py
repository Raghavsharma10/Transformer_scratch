def base_query(cls, db_session=None):
        """
        returns base query for specific service

        :param db_session:
        :return: query
        """
        db_session = get_db_session(db_session)
        return db_session.query(cls.model)