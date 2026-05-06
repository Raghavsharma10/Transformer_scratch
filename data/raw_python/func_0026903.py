def get(cls, resource_id, db_session=None):
        """
        Fetch row using primary key -
        will use existing object in session if already present

        :param resource_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        return db_session.query(cls.model).get(resource_id)