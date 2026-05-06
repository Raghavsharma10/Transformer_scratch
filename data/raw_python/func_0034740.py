def register_hook(self, hook_name, fn):
        """Register a function to be called on a GitHub event."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = fn
        else:
            raise Exception('%s hook already registered' % hook_name)