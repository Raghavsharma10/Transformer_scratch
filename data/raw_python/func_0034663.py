def _filter(request, object_, tags=None, more=False, orderby='created'):
    """Filters Piece objects from self based on filters, search, and range

    :param tags: List of tag IDs to filter
    :type tags: list
    :param more -- bool, Returns more of the same filtered set of images based on session range

    return list, Objects filtered
    """
    res = Result()

    models = QUERY_MODELS

    idDict = {}
    objDict = {}
    data = {}
    modelmap = {}
    length = 75

    # -- Get all IDs for each model
    for m in models:
        modelmap[m.model_class()] = m.model

        if object_:
            idDict[m.model] = m.model_class().objects.filter(gallery=object_)
        else:
            idDict[m.model] = m.model_class().objects.all()

        if idDict[m.model] is None:
            continue

        if tags:
            for bucket in tags:
                searchQuery = ""
                o = None
                for item in bucket:
                    if item == 0:
                        # -- filter by tagless
                        idDict[m.model].annotate(num_tags=Count('tags'))
                        if not o:
                            o = Q()
                        o |= Q(num_tags__lte=1)
                        break
                    elif isinstance(item, six.integer_types):
                        # -- filter by tag
                        if not o:
                            o = Q()
                        o |= Q(tags__id=item)
                    else:
                        # -- add to search string
                        searchQuery += item + ' '
                        if not HAYSTACK:
                            if not o:
                                o = Q()
                            # -- use a basic search
                            o |= Q(title__icontains=item)

                if HAYSTACK and searchQuery != "":
                    # -- once all tags have been filtered, filter by search
                    searchIDs = search(searchQuery, m.model_class())
                    if searchIDs:
                        if not o:
                            o = Q()
                        o |= Q(id__in=searchIDs)

                if o:
                    # -- apply the filters
                    idDict[m.model] = idDict[m.model].annotate(num_tags=Count('tags')).filter(o)
                else:
                    idDict[m.model] = idDict[m.model].none()

        # -- Get all ids of filtered objects, this will be a very fast query
        idDict[m.model] = list(idDict[m.model].order_by('-{}'.format(orderby)).values_list('id', flat=True))
        lastid = request.session.get('last_{}'.format(m.model), 0)
        if not idDict[m.model]:
            continue

        if not more:
            lastid = idDict[m.model][0]

        index = idDict[m.model].index(lastid)
        if more and lastid != 0:
            index += 1
        idDict[m.model] = idDict[m.model][index:index + length]

        # -- perform the main query to retrieve the objects we want
        objDict[m.model] = m.model_class().objects.filter(id__in=idDict[m.model])
        objDict[m.model] = objDict[m.model].select_related('author').prefetch_related('tags').order_by('-{}'.format(orderby))
        objDict[m.model] = list(objDict[m.model])

        # -- combine and sort all objects by date
    objects = _sortObjects(orderby, **objDict) if len(models) > 1 else objDict.values()[0]
    objects = objects[:length]

    # -- Find out last ids
    lastids = {}
    for obj in objects:
        lastids['last_{}'.format(modelmap[obj.__class__])] = obj.id

    for key, value in lastids.items():
        request.session[key] = value

    # -- serialize objects
    for i in objects:
        res.append(i.json())

    data['count'] = len(objects)
    if settings.DEBUG:
        data['queries'] = connection.queries

    res.value = data

    return JsonResponse(res.asDict())