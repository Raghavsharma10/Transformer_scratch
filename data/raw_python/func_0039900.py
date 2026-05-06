def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l