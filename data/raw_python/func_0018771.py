def global_add(self, key: str, value: Any) -> None:
        """
        Adds a key and value to the global dictionary
        """
        self.global_context[key] = value