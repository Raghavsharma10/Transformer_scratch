def qs_to_json(qs, fields=None):
    """
    transform QuerySet to json
    """
    if not fields :
        fields = [f.name for f in qs.model._meta.fields]


    # сформируем список для сериализации
    objects = []
    for value_dict in qs.values(*fields):
        # сохраним порядок полей, как определено в моделе
        o = OrderedDict()
        for f in fields:
            o[f] = value_dict[f]
        objects.append(o)

    # сериализуем
    json_qs = json.dumps(objects, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder)
    return json_qs