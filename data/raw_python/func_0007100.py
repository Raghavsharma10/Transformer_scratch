def modifie(self, key: str, value: Any) -> None:
        """Store the modification. `value` should be dumped in DB compatible format."""
        if key in self.FIELDS_OPTIONS:
            self.modifie_options(key, value)
        else:
            self.modifications[key] = value