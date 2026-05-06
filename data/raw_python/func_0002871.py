def insert_before(old, new):
    """
    A simple way to insert a new element node before the old element node among
    its siblings.
    """
    parent = old.getparent()
    parent.insert(parent.index(old), new)