def _has_desired_permit(permits, acategory, astatus):
    """
    return True if permits has one whose
    category_code and status_code match with the given ones
    """
    if permits is None:
        return False
    for permit in permits:
        if permit.category_code == acategory and\
           permit.status_code == astatus:
            return True
    return False