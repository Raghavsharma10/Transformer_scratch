def can_into(self, val: str) -> bool:
        """Determine if there is a leaf node with name `val`"""
        return val in self.paths or (self.param and self.param_name == val)