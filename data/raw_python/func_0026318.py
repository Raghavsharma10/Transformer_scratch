def entries(self):
        """A list of :class:`PasswordEntry` objects."""
        passwords = []
        for store in self.stores:
            passwords.extend(store.entries)
        return natsort(passwords, key=lambda e: e.name)