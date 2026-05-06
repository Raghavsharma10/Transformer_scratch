def remove_roles(self, databaseName, roleNames, collectionName=None):
        """Remove multiple roles
        
        Args:
            databaseName (str): Database Name
            roleNames (list of RoleSpecs): roles
            
        Keyword Args:
            collectionName (str): Collection
        """
        for roleName in roleNames:
            self.remove_role(databaseName, roleName, collectionName)