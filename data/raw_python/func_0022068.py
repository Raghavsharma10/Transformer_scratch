def unfollow(user, obj):
    """ Make a user unfollow an object """
    try:
        follow = Follow.objects.get_follows(obj).get(user=user)
        follow.delete()
        return follow 
    except Follow.DoesNotExist:
        pass