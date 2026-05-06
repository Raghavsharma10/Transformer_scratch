def select(cls, *args, **kwargs):
        """Support read slaves."""
        query = super(Model, cls).select(*args, **kwargs)
        query.database = cls._get_read_database()
        return query