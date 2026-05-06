def get(cls, group_id, db_session=None):
        """
        Fetch row using primary key -
        will use existing object in session if already present

        :param group_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        return db_session.query(cls.model).get(group_id)