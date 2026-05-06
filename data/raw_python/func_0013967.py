def supplement_filesystem(old_size, user_cap=False):
    """Return new size accounting for the metadata."""
    new_size = old_size
    if user_cap:
        if old_size <= _GiB_to_Byte(1.5):
            new_size = _GiB_to_Byte(3)
        else:
            new_size += _GiB_to_Byte(1.5)
    return int(new_size)