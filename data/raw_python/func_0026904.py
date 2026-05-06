def perms_for_user(cls, instance, user, db_session=None):
        """
        returns all permissions that given user has for this resource
            from groups and directly set ones too

        :param instance:
        :param user:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, instance)
        query = db_session.query(
            cls.models_proxy.GroupResourcePermission.group_id.label("owner_id"),
            cls.models_proxy.GroupResourcePermission.perm_name,
            sa.literal("group").label("type"),
        )
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.group_id.in_(
                [gr.id for gr in user.groups]
            )
        )
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.resource_id == instance.resource_id
        )

        query2 = db_session.query(
            cls.models_proxy.UserResourcePermission.user_id.label("owner_id"),
            cls.models_proxy.UserResourcePermission.perm_name,
            sa.literal("user").label("type"),
        )
        query2 = query2.filter(
            cls.models_proxy.UserResourcePermission.user_id == user.id
        )
        query2 = query2.filter(
            cls.models_proxy.UserResourcePermission.resource_id == instance.resource_id
        )
        query = query.union(query2)

        groups_dict = dict([(g.id, g) for g in user.groups])
        perms = [
            PermissionTuple(
                user,
                row.perm_name,
                row.type,
                groups_dict.get(row.owner_id) if row.type == "group" else None,
                instance,
                False,
                True,
            )
            for row in query
        ]

        # include all perms if user is the owner of this resource
        if instance.owner_user_id == user.id:
            perms.append(
                PermissionTuple(
                    user, ALL_PERMISSIONS, "user", None, instance, True, True
                )
            )
        groups_dict = dict([(g.id, g) for g in user.groups])
        if instance.owner_group_id in groups_dict:
            perms.append(
                PermissionTuple(
                    user,
                    ALL_PERMISSIONS,
                    "group",
                    groups_dict.get(instance.owner_group_id),
                    instance,
                    True,
                    True,
                )
            )

        return perms