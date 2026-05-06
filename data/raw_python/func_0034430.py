def commentToJson(comment):
    """Returns a serializable Comment dict

    :param comment: Comment to get info for
    :type comment: Comment
    :returns: dict
    """
    obj = {
        'id': comment.id,
        'comment': comment.comment,
        'user': userToJson(comment.user),
        'date': comment.submit_date.isoformat(),
    }

    return obj