def user_names_like(cls, user_name, db_session=None):
        """
        fetch users with similar names using LIKE clause

        :param user_name:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(
            sa.func.lower(cls.model.user_name).like((user_name or "").lower())
        )
        query = query.order_by(cls.model.user_name)
        # q = q.options(sa.orm.eagerload('groups'))
        return query