def add_role(self, databaseName, roleName, collectionName=None):
        """Add one role
        
        Args:
            databaseName (str): Database Name
            roleName (RoleSpecs): role
            
        Keyword Args:
            collectionName (str): Collection
            
        Raises:
            ErrRole: role not compatible with the databaseName and/or collectionName
        """
        role = {"databaseName" : databaseName,
                "roleName" : roleName}
        
        if collectionName:
            role["collectionName"] = collectionName
        
        # Check atlas constraints
        if collectionName and roleName not in [RoleSpecs.read, RoleSpecs.readWrite]:
            raise ErrRole("Permissions [%s] not available for a collection" % roleName)
        elif not collectionName and roleName not in [RoleSpecs.read, RoleSpecs.readWrite, RoleSpecs.dbAdmin] and databaseName != "admin":
            raise ErrRole("Permissions [%s] is only available for admin database" % roleName)
        
        if role not in self.roles:
            self.roles.append(role)