def validate_date(date, project_member_id, filename):
    """
    Check if date is in ISO 8601 format.

    :param date: This field is the date to be checked.
    :param project_member_id: This field is the project_member_id corresponding
        to the date provided.
    :param filename: This field is the filename corresponding to the date
        provided.
    """
    try:
        arrow.get(date)
    except Exception:
        return False
    return True