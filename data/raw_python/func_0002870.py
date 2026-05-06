def replace(old, new):
    """
    A simple way to replace one element node with another.
    """
    parent = old.getparent()
    parent.replace(old, new)