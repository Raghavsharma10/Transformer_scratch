def commentList(request):
    """Returns a rendered list of comments
    :returns: html
    """
    if request.method == 'POST':
        return post(request)

    comments = []
    guid = request.GET.get('guid', None)

    if guid:
        obj = getObjectsFromGuids([guid])[0]
        if obj.AssetType == 1:
            model = 'image'
        else:
            model = 'video'
        contenttype = ContentType.objects.get(app_label="frog", model=model)
        comments = Comment.objects.filter(object_pk=obj.id, content_type=contenttype)

    res = Result()
    for comment in comments:
        res.append(commentToJson(comment))
    return JsonResponse(res.asDict())