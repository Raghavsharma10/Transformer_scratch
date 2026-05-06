def put(request, obj_id=None):
    """ Adds Image and Video objects to Gallery based on GUIDs """
    data = request.PUT or json.loads(request.body)['body']
    guids = data.get('guids', '').split(',')
    move = data.get('from')
    security = request.PUT.get('security')
    gallery = Gallery.objects.get(pk=obj_id)
    
    if guids:
        objects = getObjectsFromGuids(guids)

        images = filter(lambda x: isinstance(x, Image), objects)
        videos = filter(lambda x: isinstance(x, Video), objects)

        gallery.images.add(*images)
        gallery.videos.add(*videos)

        if move:
            fromgallery = Gallery.objects.get(pk=move)
            fromgallery.images.remove(*images)
            fromgallery.videos.remove(*videos)
    
    if security is not None:
        gallery.security = json.loads(security)
        gallery.save()
        for child in gallery.gallery_set.all():
            child.security = gallery.security
            child.save()

    res = Result()
    res.append(gallery.json())

    return JsonResponse(res.asDict())