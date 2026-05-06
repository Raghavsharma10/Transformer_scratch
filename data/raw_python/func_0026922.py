def by_user_names(cls, user_names, db_session=None):
        """
        fetch user objects by user names

        :param user_names:
        :param db_session:
        :return:
        """
        user_names = [(name or "").lower() for name in user_names]
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(sa.func.lower(cls.model.user_name).in_(user_names))
        # q = q.options(sa.orm.eagerload(cls.groups))
        return query