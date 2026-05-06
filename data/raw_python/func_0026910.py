def groups_for_perm(
        cls,
        instance,
        perm_name,
        group_ids=None,
        limit_group_permissions=False,
        db_session=None,
    ):
        """
        return PermissionTuples for groups that have given
        permission for the resource, perm_name is __any_permission__ then
        users with any permission will be listed

        :param instance:
        :param perm_name:
        :param group_ids: limits the permissions to specific group ids
        :param limit_group_permissions: should be used if we do not want to have
        user objects returned for group permissions, this might cause performance
        issues for big groups
        :param db_session:
        :return:
        """  # noqa
        db_session = get_db_session(db_session, instance)
        group_perms = resource_permissions_for_users(
            cls.models_proxy,
            [perm_name],
            [instance.resource_id],
            group_ids=group_ids,
            limit_group_permissions=limit_group_permissions,
            skip_user_perms=True,
            db_session=db_session,
        )
        if instance.owner_group_id:
            for user in instance.owner_group.users:
                group_perms.append(
                    PermissionTuple(
                        user,
                        ALL_PERMISSIONS,
                        "group",
                        instance.owner_group,
                        instance,
                        True,
                        True,
                    )
                )

        return group_perms