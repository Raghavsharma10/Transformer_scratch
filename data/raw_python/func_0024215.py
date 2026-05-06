def add(self, name, value):
        # type: (str, str) -> None
        """Adds a new value for the given key."""
        self._last_key = name
        if name in self:
            self._dict[name] = value
            self._as_list[name].append(value)
        else:
            self[name] = value