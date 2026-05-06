def set(self, key: Any, value: Any) -> None:
        """ Sets the value of a key to a supplied value """
        if key is not None:
            self[key] = value