def channel_history(current):
    """
        Get old messages for a channel. 20 messages per request

        .. code-block:: python

            #  request:
                {
                'view':'_zops_channel_history,
                'channel_key': key,
                'timestamp': datetime, # timestamp data of oldest shown message
                }

            #  response:
                {
                'messages': [MSG_DICT, ],
                'status': 'OK',
                'code': 200
                }
    """
    current.output = {
        'status': 'OK',
        'code': 201,
        'messages': []
    }

    for msg in list(Message.objects.filter(channel_id=current.input['channel_key'],
                                      updated_at__lte=current.input['timestamp'])[:20]):
        current.output['messages'].insert(0, msg.serialize(current.user))
    # FIXME: looks like  pyoko's __lt is broken
    # TODO: convert lte to lt and remove this block, when __lt filter fixed
    if current.output['messages']:
        current.output['messages'].pop(-1)