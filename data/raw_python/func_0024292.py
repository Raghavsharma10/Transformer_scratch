def search_user(current):
    """
        Search users for adding to a public room
        or creating one to one direct messaging

        .. code-block:: python

            #  request:
                {
                'view':'_zops_search_user',
                'query': string,
                }

            #  response:
                {
                'results': [('full_name', 'key', 'avatar_url'), ],
                'status': 'OK',
                'code': 200
                }
    """
    current.output = {
        'results': [],
        'status': 'OK',
        'code': 201
    }
    qs = UserModel(current).objects.exclude(key=current.user_id).search_on(
        *settings.MESSAGING_USER_SEARCH_FIELDS,
        contains=current.input['query'])
    # FIXME: somehow exclude(key=current.user_id) not working with search_on()

    for user in qs:
        if user.key != current.user_id:
            current.output['results'].append((user.full_name, user.key, user.get_avatar_url()))