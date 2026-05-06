def from_resource_deeper(
        self, resource_id=None, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you subtree of ordered objects relative
        to the start resource_id (currently only implemented in postgresql)

        :param resource_id:
        :param limit_depth:
        :param db_session:
        :return:
        """
        return self.service.from_resource_deeper(
            resource_id=resource_id,
            limit_depth=limit_depth,
            db_session=db_session,
            *args,
            **kwargs
        )