def getSpecs(self):
        """Get specs
        
        Returns:
            dict: Representation of the object
        """
        content = {
            "databaseName" : self.databaseName,
            "roles" : self.roles,
            "username" : self.username,
            "password" : self.password
        }
        
        return content