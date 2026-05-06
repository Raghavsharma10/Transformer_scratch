def _get_least_permissions_aces(self, resources):
        """ Get ACEs with the least permissions that fit all resources.

        To have access to polymorph on N collections, user MUST have
        access to all of them. If this is true, ACEs are returned, that
        allows 'view' permissions to current request principals.

        Otherwise None is returned thus blocking all permissions except
        those defined in `nefertari.acl.BaseACL`.

        :param resources:
        :type resources: list of Resource instances
        :return: Generated Pyramid ACEs or None
        :rtype: tuple or None
        """
        factories = [res.view._factory for res in resources]
        contexts = [factory(self.request) for factory in factories]
        for ctx in contexts:
            if not self.request.has_permission('view', ctx):
                return
        else:
            return [
                (Allow, principal, 'view')
                for principal in self.request.effective_principals
            ]