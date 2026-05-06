def get_method(self, name):
        """
        Get registered method callend `name`.
        """
        try:
            return self.funcs[name]
        except KeyError:
            try:
                return self.instance._get_method(name)
            except AttributeError:
                return SimpleXMLRPCServer.resolve_dotted_attribute(
                    self.instance, name, self.allow_dotted_names)