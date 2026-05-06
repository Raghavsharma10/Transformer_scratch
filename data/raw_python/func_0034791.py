def get(request, obj_id=None):
    """Lists all tags

    :returns: json
    """
    res = Result()
    if obj_id:
        if obj_id == '0':
            obj = {
                'id': 0,
                'name': 'TAGLESS',
                'artist': False,
            }
        else:
            obj = get_object_or_404(Tag, pk=obj_id).json()

        res.append(obj)
        return JsonResponse(res.asDict())
    else:
        if request.GET.get('count'):
            itags = Tag.objects.all().annotate(icount=Count('image'))
            vtags = Tag.objects.all().annotate(vcount=Count('video'))

            for i, tag in enumerate(itags):
                tag.count = itags[i].icount + vtags[i].vcount
                res.append(tag.json())
        else:
            for tag in Tag.objects.all():
                res.append(tag.json())

        return JsonResponse(res.asDict())