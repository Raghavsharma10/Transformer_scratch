def perm_by_group_and_perm_name(
        cls, resource_id, group_id, perm_name, db_session=None
    ):
        """
        fetch permissions by group and permission name

        :param resource_id:
        :param group_id:
        :param perm_name:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.models_proxy.GroupResourcePermission)
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.group_id == group_id
        )
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.perm_name == perm_name
        )
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.resource_id == resource_id
        )
        return query.first()