def call_fget(self, obj) -> Any:
        """Return the predefined custom value when available, otherwise,
        the value defined by the getter function."""
        custom = vars(obj).get(self.name)
        if custom is None:
            return self.fget(obj)
        return custom