def resources_with_possible_perms(
        cls, instance, resource_ids=None, resource_types=None, db_session=None
    ):
        """
        returns list of permissions and resources for this user

        :param instance:
        :param resource_ids: restricts the search to specific resources
        :param resource_types: restricts the search to specific resource types
        :param db_session:
        :return:
        """
        perms = resource_permissions_for_users(
            cls.models_proxy,
            ANY_PERMISSION,
            resource_ids=resource_ids,
            resource_types=resource_types,
            user_ids=[instance.id],
            db_session=db_session,
        )
        for resource in instance.resources:
            perms.append(
                PermissionTuple(
                    instance, ALL_PERMISSIONS, "user", None, resource, True, True
                )
            )
        for group in cls.groups_with_resources(instance):
            for resource in group.resources:
                perms.append(
                    PermissionTuple(
                        instance, ALL_PERMISSIONS, "group", group, resource, True, True
                    )
                )

        return perms