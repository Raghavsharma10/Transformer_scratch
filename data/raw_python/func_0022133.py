def parse_get_params(request):
    """
    parse all url get params that contains dots in a representation of
    serializer field names, for example: d.docs.limit to d_docs_limit.
    that makes compatible an actual API client with django-rest-framework
    serializers.
    :param request:
    :return: QueryDict with parsed get params.
    """

    get = request.GET.copy()
    new_get = request.GET.copy()
    for key in get.iterkeys():
        if key.count(".") > 0:
            new_key = key.replace(".", "_")
            new_get[new_key] = get.get(key)
            del new_get[key]

    return new_get