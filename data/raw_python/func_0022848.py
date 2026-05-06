def register_hooks(self, field):
        """Register a field on its target hooks."""
        for hook, subhooks in field.register_hooks():
            self.hooks[hook].append(field)
            self.subhooks[hook] |= set(subhooks)