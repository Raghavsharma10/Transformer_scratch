def post(request):
    """ Create a Gallery """
    defaultname = 'New Gallery %i' % Gallery.objects.all().count()
    data = request.POST or json.loads(request.body)['body']
    title = data.get('title', defaultname)
    description = data.get('description', '')
    security = int(data.get('security', Gallery.PUBLIC))

    g, created = Gallery.objects.get_or_create(title=title)
    g.security = security
    g.description = description
    g.owner = request.user
    g.save()

    res = Result()
    res.append(g.json())
    res.message = 'Gallery created' if created else ''

    return JsonResponse(res.asDict())