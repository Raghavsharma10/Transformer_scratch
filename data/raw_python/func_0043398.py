def choices(klass):
    """
    Decorator to set `CHOICES` and other attributes
    """
    _choices = []
    for attr in user_attributes(klass.Meta):
        val = getattr(klass.Meta, attr)
        setattr(klass, attr, val[0])
        _choices.append((val[0], val[1]))
    setattr(klass, 'CHOICES', tuple(_choices))
    return klass