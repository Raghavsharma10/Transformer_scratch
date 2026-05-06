def toggle(user, obj):
    """ Toggles a follow status. Useful function if you don't want to perform follow
    checks but just toggle it on / off. """
    if Follow.objects.is_following(user, obj):
        return unfollow(user, obj)
    return follow(user, obj)