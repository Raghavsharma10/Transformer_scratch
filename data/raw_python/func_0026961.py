def count_children(cls, resource_id, db_session=None, *args, **kwargs):
        """
        Counts children of resource node

        :param resource_id:
        :param db_session:
        :return:
        """
        query = db_session.query(cls.model.resource_id)
        query = query.filter(cls.model.parent_id == resource_id)
        return query.count()