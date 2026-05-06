def delete_channel(current):
    """
        Delete a channel

        .. code-block:: python

            #  request:
                {
                'view':'_zops_delete_channel,
                'channel_key': key,
                }

            #  response:
                {
                'status': 'OK',
                'code': 200
                }
    """
    ch_key = current.input['channel_key']

    ch = Channel(current).objects.get(owner_id=current.user_id, key=ch_key)
    ch.delete()
    Subscriber.objects.filter(channel_id=ch_key).delete()
    Message.objects.filter(channel_id=ch_key).delete()
    current.output = {'status': 'Deleted', 'code': 200}