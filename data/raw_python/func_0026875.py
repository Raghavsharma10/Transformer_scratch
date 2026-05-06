def count_children(self, resource_id, db_session=None, *args, **kwargs):
        """
        Counts children of resource node

        :param resource_id:
        :param db_session:
        :return:
        """
        return self.service.count_children(
            resource_id=resource_id, db_session=db_session, *args, **kwargs
        )