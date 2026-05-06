def set_position(cls, resource_id, to_position, db_session=None, *args, **kwargs):
        """
        Sets node position for new node in the tree

        :param resource_id: resource to move
        :param to_position: new position
        :param db_session:
        :return:def count_children(cls, resource_id, db_session=None):
        """
        db_session = get_db_session(db_session)
        # lets lock rows to prevent bad tree states
        resource = ResourceService.lock_resource_for_update(
            resource_id=resource_id, db_session=db_session
        )
        cls.check_node_position(
            resource.parent_id, to_position, on_same_branch=True, db_session=db_session
        )
        cls.shift_ordering_up(resource.parent_id, to_position, db_session=db_session)
        db_session.flush()
        db_session.expire(resource)
        resource.ordering = to_position
        return True