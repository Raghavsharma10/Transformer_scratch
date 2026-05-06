def resources_with_perms(
        cls, instance, perms, resource_ids=None, resource_types=None, db_session=None
    ):
        """
        returns all resources that user has perms for
            (note that at least one perm needs to be met)

        :param instance:
        :param perms:
        :param resource_ids: restricts the search to specific resources
        :param resource_types:
        :param db_session:
        :return:
        """
        # owned entities have ALL permissions so we return those resources too
        # even without explict perms set
        # TODO: implement admin superrule perm - maybe return all apps
        db_session = get_db_session(db_session, instance)
        query = db_session.query(cls.models_proxy.Resource).distinct()
        group_ids = [gr.id for gr in instance.groups]
        # if user has some groups lets try to join based on their permissions
        if group_ids:
            join_conditions = (
                cls.models_proxy.GroupResourcePermission.group_id.in_(group_ids),
                cls.models_proxy.Resource.resource_id
                == cls.models_proxy.GroupResourcePermission.resource_id,
                cls.models_proxy.GroupResourcePermission.perm_name.in_(perms),
            )
            query = query.outerjoin(
                (cls.models_proxy.GroupResourcePermission, sa.and_(*join_conditions))
            )
            # ensure outerjoin permissions are correct -
            # dont add empty rows from join
            # conditions are - join ON possible group permissions
            # OR owning group/user
            query = query.filter(
                sa.or_(
                    cls.models_proxy.Resource.owner_user_id == instance.id,
                    cls.models_proxy.Resource.owner_group_id.in_(group_ids),
                    cls.models_proxy.GroupResourcePermission.perm_name != None,
                )  # noqa
            )
        else:
            # filter just by username
            query = query.filter(cls.models_proxy.Resource.owner_user_id == instance.id)
        # lets try by custom user permissions for resource
        query2 = db_session.query(cls.models_proxy.Resource).distinct()
        query2 = query2.filter(
            cls.models_proxy.UserResourcePermission.user_id == instance.id
        )
        query2 = query2.filter(
            cls.models_proxy.Resource.resource_id
            == cls.models_proxy.UserResourcePermission.resource_id
        )
        query2 = query2.filter(
            cls.models_proxy.UserResourcePermission.perm_name.in_(perms)
        )
        if resource_ids:
            query = query.filter(
                cls.models_proxy.Resource.resource_id.in_(resource_ids)
            )
            query2 = query2.filter(
                cls.models_proxy.Resource.resource_id.in_(resource_ids)
            )

        if resource_types:
            query = query.filter(
                cls.models_proxy.Resource.resource_type.in_(resource_types)
            )
            query2 = query2.filter(
                cls.models_proxy.Resource.resource_type.in_(resource_types)
            )
        query = query.union(query2)
        query = query.order_by(cls.models_proxy.Resource.resource_name)
        return query