def direct_perms_for_user(cls, instance, user, db_session=None):
        """
        returns permissions that given user has for this resource
            without ones inherited from groups that user belongs to

        :param instance:
        :param user:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, instance)
        query = db_session.query(
            cls.models_proxy.UserResourcePermission.user_id,
            cls.models_proxy.UserResourcePermission.perm_name,
        )
        query = query.filter(cls.models_proxy.UserResourcePermission.user_id == user.id)
        query = query.filter(
            cls.models_proxy.UserResourcePermission.resource_id == instance.resource_id
        )

        perms = [
            PermissionTuple(user, row.perm_name, "user", None, instance, False, True)
            for row in query
        ]

        # include all perms if user is the owner of this resource
        if instance.owner_user_id == user.id:
            perms.append(
                PermissionTuple(user, ALL_PERMISSIONS, "user", None, instance, True)
            )
        return perms