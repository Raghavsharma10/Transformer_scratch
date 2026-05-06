def by_user_name_and_security_code(cls, user_name, security_code, db_session=None):
        """
        fetch user objects by user name and security code

        :param user_name:
        :param security_code:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(
            sa.func.lower(cls.model.user_name) == (user_name or "").lower()
        )
        query = query.filter(cls.model.security_code == security_code)
        return query.first()