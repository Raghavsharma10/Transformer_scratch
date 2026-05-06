def search(request):
    """
    Search for Tag objects and returns a Result object with a list of searialize Tag
    objects.

    :param search: Append a "Search for" tag
    :type search: bool
    :param zero: Exclude Tags with no items
    :type zero: bool
    :param artist: Exclude artist tags
    :type artist: bool
    :returns: json
    """
    q = request.GET.get('q', '')
    includeSearch = request.GET.get('search', False)
    nonZero = request.GET.get('zero', False)
    excludeArtist = request.GET.get('artist', False)

    if includeSearch:
        l = [{'id': 0, 'name': 'Search for: %s' % q}]
    else:
        l = []

    query = Tag.objects.filter(name__icontains=q)

    if excludeArtist:
        query = query.exclude(artist=True)

    if nonZero:
        l += [t.json() for t in query if t.count() > 0]
    else:
        l += [t.json() for t in query]

    return JsonResponse(l, safe=False)