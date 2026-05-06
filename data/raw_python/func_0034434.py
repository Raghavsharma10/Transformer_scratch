def getObjectsFromGuids(guids):
    """Gets the model objects based on a guid list

    :param guids: Guids to get objects for
    :type guids: list
    :returns: list
    """
    guids = guids[:]
    img = list(Image.objects.filter(guid__in=guids))
    vid = list(Video.objects.filter(guid__in=guids))
    objects = img + vid
    sortedobjects = []

    if objects:
        while guids:
            for obj in iter(objects):
                if obj.guid == guids[0]:
                    sortedobjects.append(obj)
                    guids.pop(0)
                    break

    return sortedobjects