def subfield_get(self, obj, type=None):
    """
    Verbatim copy from:
    https://github.com/django/django/blob/1.9.13/django/db/models/fields/subclassing.py#L38
    """
    if obj is None:
        return self
    return obj.__dict__[self.field.name]