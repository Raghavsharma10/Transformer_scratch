def _get_formatted_feature_dependencies(data):
    """
    Takes the format of the feature_order.json in featuremodel pool.
    Creates a list of conditions in the following format:
    ]
        dict(
            name='django_productline.features.admin',
            subject='django_productline',
            ctype='after'
        )
    ]
    :param data:
    :return: list
    """
    conditions = list()
    for k, v in data.items():
        for feature in v.get('after', list()):
            conditions.append(dict(
                name=k,
                subject=feature,
                ctype='after'
            ))
        if v.get('first', False):
            conditions.append(dict(
                name=k,
                subject=None,
                ctype='first'
            ))
        if v.get('last', False):
            conditions.append(dict(
                name=k,
                subject=None,
                ctype='last'
            ))
    return conditions