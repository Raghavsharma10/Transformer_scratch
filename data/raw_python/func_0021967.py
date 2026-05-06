def get_instance(cls, state):
        """:rtype: UserStorageHandler"""
        if cls.instance is None:
            cls.instance = UserStorageHandler(state)
        return cls.instance