def create_channel(current):
    """
        Create a public channel. Can be a broadcast channel or normal chat room.

        Chat room and broadcast distinction will be made at user subscription phase.

        .. code-block:: python

            #  request:
                {
                'view':'_zops_create_channel',
                'name': string,
                'description': string,
                }

            #  response:
                {
                'description': string,
                'name': string,
                'no_of_members': int,
                'member_list': [
                    {'name': string,
                     'is_online': bool,
                     'avatar_url': string,
                    }],
                'last_messages': [MSG_DICT]
                'status': 'Created',
                'code': 201,
                'key': key, # of just created channel
                }
    """
    channel = Channel(name=current.input['name'],
                      description=current.input['description'],
                      owner=current.user,
                      typ=15).save()
    with BlockSave(Subscriber):
        Subscriber.objects.get_or_create(user=channel.owner,
                                         channel=channel,
                                         can_manage=True,
                                         can_leave=False)
    current.input['key'] = channel.key
    show_channel(current)
    current.output.update({
        'status': 'Created',
        'code': 201
    })