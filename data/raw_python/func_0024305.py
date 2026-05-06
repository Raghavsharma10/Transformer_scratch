def remove_from_favorites(current):
    """
    Remove a message from favorites

    .. code-block:: python

        #  request:
            {
            'view':'_zops_remove_from_favorites,
            'key': key,
            }

        #  response:
            {
            'status': 'OK',
            'code': 200
            }

    """
    try:
        current.output = {'status': 'OK', 'code': 200}
        Favorite(current).objects.get(user_id=current.user_id,
                                      key=current.input['key']).delete()
    except ObjectDoesNotExist:
        raise HTTPError(404, "")