def path_upper(
        self, object_id, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you path to root node starting from object_id
            currently only for postgresql

        :param object_id:
        :param limit_depth:
        :param db_session:
        :return:
        """
        return self.service.path_upper(
            object_id=object_id,
            limit_depth=limit_depth,
            db_session=db_session,
            *args,
            **kwargs
        )