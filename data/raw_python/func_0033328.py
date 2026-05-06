def getSpecs(self):
        """Get specs
        
        Returns:
            dict: Representation of the object
        """
        
        content = {}
        
        if len(self.roles) != 0:
            content["roles"] = self.roles
        
        if self.password:
            content["password"] = self.password
        
        return content