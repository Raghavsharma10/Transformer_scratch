def by_email(cls, email, db_session=None):
        """
        fetch user object by email

        :param email:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model).filter(
            sa.func.lower(cls.model.email) == (email or "").lower()
        )
        query = query.options(sa.orm.eagerload("groups"))
        return query.first()