def check_node_position(
        cls, parent_id, position, on_same_branch, db_session=None, *args, **kwargs
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
        db_session = get_db_session(db_session)
        if not position or position < 1:
            raise ZigguratResourceOutOfBoundaryException(
                "Position is lower than {}", value=1
            )
        item_count = cls.count_children(parent_id, db_session=db_session)
        max_value = item_count if on_same_branch else item_count + 1
        if position > max_value:
            raise ZigguratResourceOutOfBoundaryException(
                "Maximum resource ordering is {}", value=max_value
            )