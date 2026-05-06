def lock_resource_for_update(cls, resource_id, db_session):
        """
        Selects resource for update - locking access for other transactions

        :param resource_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(cls.model.resource_id == resource_id)
        query = query.with_for_update()
        return query.first()