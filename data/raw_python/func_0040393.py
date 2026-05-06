def create_module(self, spec):
        """Improve python2 semantics for module creation."""
        mod = super(NamespaceLoader2, self).create_module(spec)
        # Set a few properties required by PEP 302
        # mod.__file__ = [p for p in self.path]
        # this will set mod.__repr__ to not builtin... shouldnt break anything in py2...
        # CAREFUL : get_filename present implies the module has ONE location, which is not true with namespaces
        return mod