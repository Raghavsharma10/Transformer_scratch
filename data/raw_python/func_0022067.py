def follow(user, obj):
    """ Make a user follow an object """
    follow, created = Follow.objects.get_or_create(user, obj)
    return follow