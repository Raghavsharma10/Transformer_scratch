def by_group_name(cls, group_name, db_session=None):
        """
        fetch group by name

        :param group_name:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model).filter(cls.model.group_name == group_name)
        return query.first()