def resources_with_possible_perms(
        cls,
        instance,
        perm_names=None,
        resource_ids=None,
        resource_types=None,
        db_session=None,
    ):
        """
        returns list of permissions and resources for this group,
            resource_ids restricts the search to specific resources

        :param instance:
        :param perm_names:
        :param resource_ids:
        :param resource_types:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, instance)

        query = db_session.query(
            cls.models_proxy.GroupResourcePermission.perm_name,
            cls.models_proxy.Group,
            cls.models_proxy.Resource,
        )
        query = query.filter(
            cls.models_proxy.Resource.resource_id
            == cls.models_proxy.GroupResourcePermission.resource_id
        )
        query = query.filter(
            cls.models_proxy.Group.id
            == cls.models_proxy.GroupResourcePermission.group_id
        )
        if resource_ids:
            query = query.filter(
                cls.models_proxy.GroupResourcePermission.resource_id.in_(resource_ids)
            )

        if resource_types:
            query = query.filter(
                cls.models_proxy.Resource.resource_type.in_(resource_types)
            )

        if perm_names not in ([ANY_PERMISSION], ANY_PERMISSION) and perm_names:
            query = query.filter(
                cls.models_proxy.GroupResourcePermission.perm_name.in_(perm_names)
            )
        query = query.filter(
            cls.models_proxy.GroupResourcePermission.group_id == instance.id
        )

        perms = [
            PermissionTuple(
                None, row.perm_name, "group", instance, row.Resource, False, True
            )
            for row in query
        ]
        for resource in instance.resources:
            perms.append(
                PermissionTuple(
                    None, ALL_PERMISSIONS, "group", instance, resource, True, True
                )
            )
        return perms