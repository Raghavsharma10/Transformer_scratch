def has_permissions(self, object, group, operations):
        '''Check if this :class:`Subject` has permissions for ``operations``
on an ``object``. It returns the number of valid permissions.'''
        if self.is_superuser:
            return 1
        else:
            models = self.session.router
            # valid permissions
            query = models.permission.for_object(object, operation=operations)
            objects = models[models.role.permissions.model]
            return objects.filter(role=self.role.query(),
                                  permission=query).count()