def add_to_favorites(current):
    """
    Favorite a message

    .. code-block:: python

        #  request:
            {
            'view':'_zops_add_to_favorites,
            'key': key,
            }

        #  response:
            {
            'status': 'Created',
            'code': 201
            'favorite_key': key
            }

    """
    msg = Message.objects.get(current.input['key'])
    current.output = {'status': 'Created', 'code': 201}
    fav, new = Favorite.objects.get_or_create(user_id=current.user_id, message=msg)
    current.output['favorite_key'] = fav.key