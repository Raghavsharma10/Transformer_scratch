def move_to_position(
        self,
        resource_id,
        to_position,
        new_parent_id=noop,
        db_session=None,
        *args,
        **kwargs
    ):
        """
        Moves node to new location in the tree

        :param resource_id: resource to move
        :param to_position: new position
        :param new_parent_id: new parent id
        :param db_session:
        :return:
        """
        return self.service.move_to_position(
            resource_id=resource_id,
            to_position=to_position,
            new_parent_id=new_parent_id,
            db_session=db_session,
            *args,
            **kwargs
        )