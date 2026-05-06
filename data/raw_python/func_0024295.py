def find_message(current):
    """
        Search in messages. If "channel_key" given, search will be limited to that channel,
        otherwise search will be performed on all of user's subscribed channels.

        .. code-block:: python

            #  request:
                {
                'view':'_zops_search_unit,
                'channel_key': key,
                'query': string,
                'page': int,
                }

            #  response:
                {
                'results': [MSG_DICT, ],
                'pagination': {
                    'page': int, # current page
                    'total_pages': int,
                    'total_objects': int,
                    'per_page': int, # object per page
                    },
                'status': 'OK',
                'code': 200
                }
    """
    current.output = {
        'results': [],
        'status': 'OK',
        'code': 201
    }
    query_set = Message(current).objects.search_on(['msg_title', 'body', 'url'],
                                                   contains=current.input['query'])
    if current.input['channel_key']:
        query_set = query_set.filter(channel_id=current.input['channel_key'])
    else:
        subscribed_channels = Subscriber.objects.filter(user_id=current.user_id).values_list(
            "channel_id", flatten=True)
        query_set = query_set.filter(channel_id__in=subscribed_channels)

    query_set, pagination_data = _paginate(current_page=current.input['page'], query_set=query_set)
    current.output['pagination'] = pagination_data
    for msg in query_set:
        current.output['results'].append(msg.serialize(current.user))