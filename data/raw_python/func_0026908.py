def by_resource_id(cls, resource_id, db_session=None):
        """
        fetch the resouce by id

        :param resource_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model).filter(
            cls.model.resource_id == int(resource_id)
        )
        return query.first()