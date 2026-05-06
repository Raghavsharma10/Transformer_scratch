def serialize_checks(check_set):
    """
    Serialize a check_set for raphael
    """
    check_set_list = []
    for check in check_set.all()[:25]:
        check_set_list.append(
            {
                'datetime': check.checked_datetime.isoformat(),
                'value': check.response_time,
                'success': 1 if check.success else 0
            }
        )
    return check_set_list