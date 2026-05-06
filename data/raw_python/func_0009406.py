def next_minor(self):
    """
    Return a Version whose minor number is one greater than self's.

    .. note::
        The new Version will always have a zeroed-out bugfix/tertiary version
        number, because the "next minor release" of e.g. 1.2.1 is 1.3.0, not
        1.3.1.
    """
    clone = self.clone()
    clone.minor += 1
    clone.patch = 0
    return clone