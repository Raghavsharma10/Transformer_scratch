def contains_list(longer, shorter):
    """Check if longer list starts with shorter list"""
    if len(longer) <= len(shorter):
        return False
    for a, b in zip(shorter, longer):
        if a != b:
            return False
    return True