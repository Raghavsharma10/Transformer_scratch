def override(self, value):
        """Temporarily overrides the old value with the new one."""
        if self._value is not value:
            return _ScopedValueOverrideContext(self, value)
        else:
            return empty_context