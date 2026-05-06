def create_direct_channel(current):
    """
    Create a One-To-One channel between current and selected user.


    .. code-block:: python

        #  request:
            {
            'view':'_zops_create_direct_channel',
            'user_key': key,
            }

        #  response:
            {
            'description': string,
            'no_of_members': int,
            'member_list': [
                {'name': string,
                 'is_online': bool,
                 'avatar_url': string,
                }],
            'last_messages': [MSG_DICT]
            'status': 'Created',
            'code': 201,
            'channel_key': key, # of just created channel
            'name': string, # name of subscribed channel
            }
    """
    channel, sub_name = Channel.get_or_create_direct_channel(current.user_id,
                                                             current.input['user_key'])
    current.input['key'] = channel.key
    show_channel(current)
    current.output.update({
        'status': 'Created',
        'code': 201
    })