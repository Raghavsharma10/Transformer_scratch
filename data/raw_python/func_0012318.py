def new(cls, password, rounds):
        """Creates a PasswordHash from the given password."""
        if isinstance(password, str):
            password = password.encode('utf-8')
        return cls(cls._new(password, rounds))