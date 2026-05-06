def delete_branch(self, resource_id=None, db_session=None, *args, **kwargs):
        """
        This deletes whole branch with children starting from resource_id

        :param resource_id:
        :param db_session:
        :return:
        """
        return self.service.delete_branch(
            resource_id=resource_id, db_session=db_session, *args, **kwargs
        )