def all(cls, klass, db_session=None):
        """
        returns all objects of specific type - will work correctly with
        sqlalchemy inheritance models, you should normally use models
        base_query()  instead of this function its for bw. compat purposes

        :param klass:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        return db_session.query(klass)