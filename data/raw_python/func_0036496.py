def rating_score(obj, user):
    """
    Returns the score a user has given an object
    """
    if not user.is_authenticated() or not hasattr(obj, '_ratings_field'):
        return False

    ratings_descriptor = getattr(obj, obj._ratings_field)
    try:
        rating = ratings_descriptor.get(user=user).score
    except ratings_descriptor.model.DoesNotExist:
        rating = None

    return rating