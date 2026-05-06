def _get_condition_instances(data):
    """
    Returns a list of OrderingCondition instances created from the passed data structure.
    The structure should be a list of dicts containing the necessary information:
    [
        dict(
            name='featureA',
            subject='featureB',
            ctype='after'
        ),
    ]
    Example says: featureA needs to be after featureB.
    :param data:
    :return:
    """
    conditions = list()
    for cond in data:
        conditions.append(OrderingCondition(
            name=cond.get('name'),
            subject=cond.get('subject'),
            ctype=cond.get('ctype')
        ))
    return conditions