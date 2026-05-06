def show_channel(current, waited=False):
    """
    Initial display of channel content.
    Returns channel description, members, no of members, last 20 messages etc.


    .. code-block:: python

        #  request:
            {
                'view':'_zops_show_channel',
                'key': key,
            }

        #  response:
            {
            'channel_key': key,
            'description': string,
            'no_of_members': int,
            'member_list': [
                {'name': string,
                 'is_online': bool,
                 'avatar_url': string,
                }],
            'name': string,
            'last_messages': [MSG_DICT]
            'status': 'OK',
            'code': 200
            }
    """
    ch = Channel(current).objects.get(current.input['key'])
    sbs = ch.get_subscription_for_user(current.user_id)
    current.output = {'key': current.input['key'],
                      'description': ch.description,
                      'name': sbs.name,
                      'actions': sbs.get_actions(),
                      'avatar_url': ch.get_avatar(current.user),
                      'no_of_members': len(ch.subscriber_set),
                      'member_list': [{'name': sb.user.full_name,
                                       'is_online': sb.user.is_online(),
                                       'avatar_url': sb.user.get_avatar_url()
                                       } for sb in ch.subscriber_set.objects.all()],
                      'last_messages': [],
                      'status': 'OK',
                      'code': 200
                      }
    for msg in ch.get_last_messages():
        current.output['last_messages'].insert(0, msg.serialize(current.user))