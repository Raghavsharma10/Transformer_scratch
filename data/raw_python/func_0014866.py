def mongoqs_to_json(qs, fields=None):
    """
    transform mongoengine.QuerySet to json
    """

    l = list(qs.as_pymongo())

    for element in l:
        element.pop('_cls')

    # use DjangoJSONEncoder for transform date data type to datetime
    json_qs = json.dumps(l, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder)
    return json_qs