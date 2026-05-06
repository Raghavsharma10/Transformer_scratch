def delete(request, obj_id=None):
    """Removes tags from objects resolved from guids

    :param tags: Tags to remove
    :type tags: list
    :param guids: Guids to remove tags from
    :type guids: list
    :returns: json
    """
    res = Result()

    if obj_id:
        # -- Delete the tag itself
        tag = Tag.objects.get(pk=obj_id)
        guids = []
        images = Image.objects.filter(tags__id=obj_id)
        guids += [_.guid for _ in images]
        videos = Video.objects.filter(tags__id=obj_id)
        guids += [_.guid for _ in videos]
        # -- Remove all tags from objects
        _manageTags([tag.id], guids, add=False)
        # -- Delete old tags
        tag.delete()
    else:
        tags = [_ for _ in request.DELETE.get('tags', '').split(',') if _]
        guids = [_ for _ in request.DELETE.get('guids', '').split(',') if _]

        _manageTags(tags, guids, add=False)

    return JsonResponse(res.asDict())