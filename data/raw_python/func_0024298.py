def pin_channel(current):
    """
        Pin a channel to top of channel list

        .. code-block:: python

            #  request:
                {
                'view':'_zops_pin_channel,
                'channel_key': key,
                }

            #  response:
                {
                'status': 'OK',
                'code': 200
                }
    """
    try:
        Subscriber(current).objects.filter(user_id=current.user_id,
                                           channel_id=current.input['channel_key']).update(
            pinned=True)
        current.output = {'status': 'OK', 'code': 200}
    except ObjectDoesNotExist:
        raise HTTPError(404, "")