def increment(self, key: Any, by: int = 1) -> None:
        """ Increments the value set against a key.  If the key is not present, 0 is assumed as the initial state """
        if key is not None:
            self[key] = self.get(key, 0) + by