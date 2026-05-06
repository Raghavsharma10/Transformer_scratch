def add_members(current):
    """
        Subscribe member(s) to a channel

        .. code-block:: python

            #  request:
                {
                'view':'_zops_add_members',
                'channel_key': key,
                'read_only': boolean, # true if this is a Broadcast channel,
                                      # false if it's a normal chat room
                'members': [key, key],
                }

            #  response:
                {
                'existing': [key,], # existing members
                'newly_added': [key,], # newly added members
                'status': 'Created',
                'code': 201
                }
    """
    newly_added, existing = [], []
    read_only = current.input['read_only']
    for member_key in current.input['members']:
        sb, new = Subscriber(current).objects.get_or_create(user_id=member_key,
                                                            read_only=read_only,
                                                            channel_id=current.input['channel_key'])
        if new:
            newly_added.append(member_key)
        else:
            existing.append(member_key)

    current.output = {
        'existing': existing,
        'newly_added': newly_added,
        'status': 'OK',
        'code': 201
    }