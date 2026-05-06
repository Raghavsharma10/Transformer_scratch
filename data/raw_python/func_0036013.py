def pop(self, name):
        """Get and remove key from database (atomic)."""
        name = mkey(name)
        temp = mkey((name, "__poptmp__"))
        self.rename(name, temp)
        value = self[temp]
        del(self[temp])
        return value