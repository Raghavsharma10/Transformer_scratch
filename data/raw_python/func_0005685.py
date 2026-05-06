def place(self, part: str, val: Union['ApiNode', 'ApiEndpoint']):
        """place a leaf node"""
        if part.startswith(':'):
            if self.param and self.param != part:
                err = """Cannot place param '{}' as '{self.param_name}' exist on node already!"""
                raise ParamAlreadyExist(err.format(part, self=self))
            self.param = val
            self.param_name = part
            return val
        self.paths[part] = val
        return val