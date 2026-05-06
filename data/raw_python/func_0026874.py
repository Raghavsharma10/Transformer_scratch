def check_node_parent(
        self, resource_id, new_parent_id, db_session=None, *args, **kwargs
    ):
        """
        Checks if parent destination is valid for node

        :param resource_id:
        :param new_parent_id:
        :param db_session:
        :return:
        """
        return self.service.check_node_parent(
            resource_id=resource_id,
            new_parent_id=new_parent_id,
            db_session=db_session,
            *args,
            **kwargs
        )