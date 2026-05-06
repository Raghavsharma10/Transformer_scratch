def permitted_query(self, query, group, operations):
        '''Change the ``query`` so that only instances for which
``group`` has roles with permission on ``operations`` are returned.'''
        session = query.session
        models = session.router
        user = group.user
        if user.is_superuser:   # super-users have all permissions
            return query
        roles = group.roles.query()
        roles = group.roles.query()  # query on all roles for group
        # The throgh model for Role/Permission relationship
        throgh_model = models.role.permissions.model
        models[throgh_model].filter(role=roles,
                                    permission__model_type=query.model,
                                    permission__operations=operations)

        # query on all relevant permissions
        permissions = router.permission.filter(model_type=query.model,
                                               level=operations)

        owner_query = query.filter(user=user)
        # all roles for the query model with appropriate permission level
        roles = models.role.filter(model_type=query.model, level__ge=level)
        # Now we need groups which have these roles
        groups = Role.groups.throughquery(
            session).filter(role=roles).get_field('group')
        # I need to know if user is in any of these groups
        if user.groups.filter(id=groups).count():
            # it is, lets get the model with permissions less
            # or equal permission level
            permitted = models.instancerole.filter(
                role=roles).get_field('object_id')
            return owner_query.union(model.objects.filter(id=permitted))
        else:
            return owner_query