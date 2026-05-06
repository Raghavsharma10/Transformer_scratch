def get(request):
    """Gets the currently logged in users preferences

    :returns: json
    """
    res = Result()
    obj, created = UserPref.objects.get_or_create(user=request.user, defaults={'data': json.dumps(DefaultPrefs.copy())})

    data = obj.json()
    data['subscriptions'] = [_.json() for _ in GallerySubscription.objects.filter(user=request.user)]

    res.append(data)

    return JsonResponse(res.asDict())