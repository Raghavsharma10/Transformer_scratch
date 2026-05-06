def by_id(cls, user_id, db_session=None):
        """
        fetch user by user id

        :param user_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(cls.model.id == user_id)
        query = query.options(sa.orm.eagerload("groups"))
        return query.first()