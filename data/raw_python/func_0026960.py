def check_node_parent(
        cls, resource_id, new_parent_id, db_session=None, *args, **kwargs
    ):
        """
        Checks if parent destination is valid for node

        :param resource_id:
        :param new_parent_id:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        new_parent = ResourceService.lock_resource_for_update(
            resource_id=new_parent_id, db_session=db_session
        )
        # we are not moving to "root" so parent should be found
        if not new_parent and new_parent_id is not None:
            raise ZigguratResourceTreeMissingException("New parent node not found")
        else:
            result = cls.path_upper(new_parent_id, db_session=db_session)
            path_ids = [r.resource_id for r in result]
            if resource_id in path_ids:
                raise ZigguratResourceTreePathException(
                    "Trying to insert node into itself"
                )