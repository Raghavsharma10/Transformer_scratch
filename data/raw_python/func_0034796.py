def merge(request, obj_id):
    """Merges multiple tags into a single tag and all related objects are reassigned"""
    res = Result()
    if request.POST:
        tags = json.loads(request.POST['tags'])
    else:
        tags = json.loads(request.body)['body']['tags']

    guids = []
    images = Image.objects.filter(tags__id__in=tags)
    guids += [_.guid for _ in images]
    videos = Video.objects.filter(tags__id__in=tags)
    guids += [_.guid for _ in videos]
    # -- Remove all tags from objects
    _manageTags(tags, guids, add=False)
    # -- Add merged tag to all objects
    _manageTags([obj_id], guids, add=True)
    # -- Delete old tags
    Tag.objects.filter(pk__in=tags).delete()

    return JsonResponse(res.asDict())