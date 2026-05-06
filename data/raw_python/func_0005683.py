def into(self, val: str) -> Union['ApiNode', 'ApiEndpoint']:
        """Get another leaf node with name `val` if possible"""
        if val in self.paths:
            return self.paths[val]
        if self.param:
            return self.param
        raise IndexError(_("Value {} is missing from api").format(val))