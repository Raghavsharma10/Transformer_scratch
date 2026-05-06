def users_for_perm(
        cls,
        instance,
        perm_name,
        user_ids=None,
        group_ids=None,
        limit_group_permissions=False,
        skip_group_perms=False,
        db_session=None,
    ):
        """
        return PermissionTuples for users AND groups that have given
        permission for the resource, perm_name is __any_permission__ then
        users with any permission will be listed

        :param instance:
        :param perm_name:
        :param user_ids: limits the permissions to specific user ids
        :param group_ids: limits the permissions to specific group ids
        :param limit_group_permissions: should be used if we do not want to have
        user objects returned for group permissions, this might cause performance
        issues for big groups
        :param skip_group_perms: do not attach group permissions to the resultset
        :param db_session:
        :return:
        """  # noqa
        db_session = get_db_session(db_session, instance)
        users_perms = resource_permissions_for_users(
            cls.models_proxy,
            [perm_name],
            [instance.resource_id],
            user_ids=user_ids,
            group_ids=group_ids,
            limit_group_permissions=limit_group_permissions,
            skip_group_perms=skip_group_perms,
            db_session=db_session,
        )
        if instance.owner_user_id:
            users_perms.append(
                PermissionTuple(
                    instance.owner, ALL_PERMISSIONS, "user", None, instance, True, True
                )
            )
        if instance.owner_group_id and not skip_group_perms:
            for user in instance.owner_group.users:
                users_perms.append(
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

        return users_perms