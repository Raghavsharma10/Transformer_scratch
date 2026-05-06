def create(cls, id=None, **kwargs):
        """
        Similar to new() however this calls save() on the object before
        returning it.
        """
        what = cls(**kwargs)
        if id:
            setattr(what, cls.primaryKey, id)
        what.save()
        return what