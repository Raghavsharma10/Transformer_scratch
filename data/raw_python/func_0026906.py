def group_perms_for_user(cls, instance, user, db_session=None):
        """
        returns permissions that given user has for this resource
            that are inherited from groups

        :param instance:
        :param user:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, instance)
        perms = resource_permissions_for_users(
            cls.models_proxy,
            ANY_PERMISSION,
            resource_ids=[instance.resource_id],
            user_ids=[user.id],
            db_session=db_session,
        )
        perms = [p for p in perms if p.type == "group"]
        # include all perms if user is the owner of this resource
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