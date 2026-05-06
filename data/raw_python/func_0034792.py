def post(request):
    """Creates a tag object

    :param name: Name for tag
    :type name: str
    :returns: json
    """
    res = Result()
    data = request.POST or json.loads(request.body)['body']
    name = data.get('name', None)

    if not name:
        res.isError = True
        res.message = "No name given"

        return JsonResponse(res.asDict())
    
    tag = Tag.objects.get_or_create(name=name.lower())[0]

    res.append(tag.json())

    return JsonResponse(res.asDict())