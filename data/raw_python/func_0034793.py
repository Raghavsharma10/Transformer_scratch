def put(request, obj_id=None):
    """Adds tags from objects resolved from guids

    :param tags: Tags to add
    :type tags: list
    :param guids: Guids to add tags from
    :type guids: list
    :returns: json
    """
    res = Result()
    data = request.PUT or json.loads(request.body)['body']
    if obj_id:
        # -- Edit the tag
        tag = Tag.objects.get(pk=obj_id)
        tag.name = data.get('name', tag.name)
        tag.artist = data.get('artist', tag.artist)
        tag.save()
    else:
        tags = [_ for _ in data.get('tags', '').split(',') if _]
        guids = [_ for _ in data.get('guids', '').split(',') if _]

        _manageTags(tags, guids)

    return JsonResponse(res.asDict())