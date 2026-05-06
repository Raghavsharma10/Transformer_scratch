def get(cls, user_id, db_session=None):
        """
        Fetch row using primary key -
        will use existing object in session if already present

        :param user_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        return db_session.query(cls.model).get(user_id)