def new(cls, *args, **kwargs):
        """Create a new instance of this model based on its spec and either
        a map or the provided kwargs."""
        new = cls(make_default(getattr(cls, 'spec', {})))
        new.update(args[0] if args and not kwargs else kwargs)
        return new