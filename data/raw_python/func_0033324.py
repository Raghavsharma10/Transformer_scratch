def add_roles(self, databaseName, roleNames, collectionName=None):
        """Add multiple roles
        
        Args:
            databaseName (str): Database Name
            roleNames (list of RoleSpecs): roles
            
        Keyword Args:
            collectionName (str): Collection
        
        Raises:
            ErrRoleException: role not compatible with the databaseName and/or collectionName
        """
        for roleName in roleNames:
            self.add_role(databaseName, roleName, collectionName)