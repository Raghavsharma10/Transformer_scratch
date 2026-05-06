def set_collections_acl(self):
        """ Calculate and set ACL valid for requested collections.

        DENY_ALL is added to ACL to make sure no access rules are
        inherited.
        """
        acl = [(Allow, 'g:admin', ALL_PERMISSIONS)]
        collections = self.get_collections()
        resources = self.get_resources(collections)
        aces = self._get_least_permissions_aces(resources)
        if aces is not None:
            for ace in aces:
                acl.append(ace)
        acl.append(DENY_ALL)
        self.__acl__ = tuple(acl)