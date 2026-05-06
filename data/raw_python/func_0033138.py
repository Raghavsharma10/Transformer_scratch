def add_user(self, name, password=None, read_only=None, db=None, **kwargs):
        """ Adds a user that can be used for authentication """
        if db is None:
            return self.get_connection().admin.add_user(
                    name, password=password, read_only=read_only, **kwargs)
        return self.get_connection()[db].add_user(
                    name, password=password, read_only=read_only, **kwargs)