def is_uuid(u):
    """validator for plumbum prompt"""
    if isinstance(u, str) and u.replace('-', '') == uuid.UUID(u).hex:
        return u
    return False