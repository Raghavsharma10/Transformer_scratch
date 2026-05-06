def move_to_position(
        cls,
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
        db_session = get_db_session(db_session)
        # lets lock rows to prevent bad tree states
        resource = ResourceService.lock_resource_for_update(
            resource_id=resource_id, db_session=db_session
        )
        ResourceService.lock_resource_for_update(
            resource_id=resource.parent_id, db_session=db_session
        )
        same_branch = False

        # reset if parent is same as old
        if new_parent_id == resource.parent_id:
            new_parent_id = noop

        if new_parent_id is not noop:
            cls.check_node_parent(resource_id, new_parent_id, db_session=db_session)
        else:
            same_branch = True

        if new_parent_id is noop:
            # it is not guaranteed that parent exists
            parent_id = resource.parent_id if resource else None
        else:
            parent_id = new_parent_id

        cls.check_node_position(
            parent_id, to_position, on_same_branch=same_branch, db_session=db_session
        )
        # move on same branch
        if new_parent_id is noop:
            order_range = list(sorted((resource.ordering, to_position)))
            move_down = resource.ordering > to_position

            query = db_session.query(cls.model)
            query = query.filter(cls.model.parent_id == parent_id)
            query = query.filter(cls.model.ordering.between(*order_range))
            if move_down:
                query.update(
                    {cls.model.ordering: cls.model.ordering + 1},
                    synchronize_session=False,
                )
            else:
                query.update(
                    {cls.model.ordering: cls.model.ordering - 1},
                    synchronize_session=False,
                )
            db_session.flush()
            db_session.expire(resource)
            resource.ordering = to_position
        # move between branches
        else:
            cls.shift_ordering_down(
                resource.parent_id, resource.ordering, db_session=db_session
            )
            cls.shift_ordering_up(new_parent_id, to_position, db_session=db_session)
            db_session.expire(resource)
            resource.parent_id = new_parent_id
            resource.ordering = to_position
            db_session.flush()
        return True