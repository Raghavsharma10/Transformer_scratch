def permissions(cls, instance, db_session=None):
        """
        returns all non-resource permissions based on what groups user
            belongs and directly set ones for this user

        :param instance:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, instance)
        query = db_session.query(
            cls.models_proxy.GroupPermission.group_id.label("owner_id"),
            cls.models_proxy.GroupPermission.perm_name.label("perm_name"),
            sa.literal("group").label("type"),
        )
        query = query.filter(
            cls.models_proxy.GroupPermission.group_id
            == cls.models_proxy.UserGroup.group_id
        )
        query = query.filter(
            cls.models_proxy.User.id == cls.models_proxy.UserGroup.user_id
        )
        query = query.filter(cls.models_proxy.User.id == instance.id)

        query2 = db_session.query(
            cls.models_proxy.UserPermission.user_id.label("owner_id"),
            cls.models_proxy.UserPermission.perm_name.label("perm_name"),
            sa.literal("user").label("type"),
        )
        query2 = query2.filter(cls.models_proxy.UserPermission.user_id == instance.id)
        query = query.union(query2)
        groups_dict = dict([(g.id, g) for g in instance.groups])
        return [
            PermissionTuple(
                instance,
                row.perm_name,
                row.type,
                groups_dict.get(row.owner_id) if row.type == "group" else None,
                None,
                False,
                True,
            )
            for row in query
        ]