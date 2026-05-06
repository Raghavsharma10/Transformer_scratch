def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))