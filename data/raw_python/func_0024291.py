def add_unit_to_channel(current):
    """
        Subscribe users of a given unit to given channel

        JSON API:
            .. code-block:: python

                #  request:
                    {
                    'view':'_zops_add_unit_to_channel',
                    'unit_key': key,
                    'channel_key': key,
                    'read_only': boolean, # true if this is a Broadcast channel,
                                          # false if it's a normal chat room

                    }

                #  response:
                    {
                    'existing': [key,], # existing members
                    'newly_added': [key,], # newly added members
                    'status': 'Created',
                    'code': 201
                    }
    """
    read_only = current.input['read_only']
    newly_added, existing = [], []
    for member_key in UnitModel.get_user_keys(current, current.input['unit_key']):
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