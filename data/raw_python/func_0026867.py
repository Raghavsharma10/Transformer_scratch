def from_parent_deeper(
        self, parent_id=None, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you subtree of ordered objects relative
        to the start parent_id (currently only implemented in postgresql)

        :param resource_id:
        :param limit_depth:
        :param db_session:
        :return:
        """
        return self.service.from_parent_deeper(
            parent_id=parent_id,
            limit_depth=limit_depth,
            db_session=db_session,
            *args,
            **kwargs
        )