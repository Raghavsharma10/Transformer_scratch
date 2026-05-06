def get(cls, external_id, local_user_id, provider_name, db_session=None):
        """
        Fetch row using primary key -
        will use existing object in session if already present

        :param external_id:
        :param local_user_id:
        :param provider_name:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        return db_session.query(cls.model).get(
            [external_id, local_user_id, provider_name]
        )