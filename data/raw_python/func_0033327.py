def remove_role(self, databaseName, roleName, collectionName=None):
        """Remove one role
        
        Args:
            databaseName (str): Database Name
            roleName (RoleSpecs): role
            
        Keyword Args:
            collectionName (str): Collection
        """
        role = {"databaseName" : databaseName,
                "roleName" : roleName}
        
        if collectionName:
            role["collectionName"] = collectionName
        
        if role in self.roles:
            self.roles.remove(role)