def get_child(parent, child_index):
    """
    Get the child at the given index, or return None if it doesn't exist.
    """
    if child_index < 0 or child_index >= len(parent.childNodes):
        return None
    return parent.childNodes[child_index]