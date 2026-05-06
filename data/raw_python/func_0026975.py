def get_one(self, context, name):
        """
        Returns a function if it is registered, the context is ignored.
        """
        try:
            return self.functions[name]
        except KeyError:
            raise FunctionNotFound(name)