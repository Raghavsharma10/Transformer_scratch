def list_favorites(current):
    """
    List user's favorites. If "channel_key" given, will return favorites belong to that channel.

    .. code-block:: python

        #  request:
            {
            'view':'_zops_list_favorites,
            'channel_key': key,
            }

        #  response:
            {
            'status': 'OK',
            'code': 200
            'favorites':[{'key': key,
                        'channel_key': key,
                        'message_key': key,
                        'message_summary': string, # max 60 char
                        'channel_name': string,
                        },]
            }

    """
    current.output = {'status': 'OK', 'code': 200, 'favorites': []}
    query_set = Favorite(current).objects.filter(user_id=current.user_id)
    if current.input['channel_key']:
        query_set = query_set.filter(channel_id=current.input['channel_key'])
    current.output['favorites'] = [{
                                       'key': fav.key,
                                       'channel_key': fav.channel.key,
                                       'message_key': fav.message.key,
                                       'message_summary': fav.summary,
                                       'channel_name': fav.channel_name
                                   } for fav in query_set]