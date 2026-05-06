def check_node_position(
        self, parent_id, position, on_same_branch, db_session=None, *args, **kwargs
    ):
        """
        Checks if node position for given parent is valid, raises exception if
        this is not the case

        :param parent_id:
        :param position:
        :param on_same_branch: indicates that we are checking same branch
        :param db_session:
        :return:
        """
        return self.service.check_node_position(
            parent_id=parent_id,
            position=position,
            on_same_branch=on_same_branch,
            db_session=db_session,
            *args,
            **kwargs
        )