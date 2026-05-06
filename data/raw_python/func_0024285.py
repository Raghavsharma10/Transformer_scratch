def report_last_seen_message(current):
    """
    Push timestamp of latest message of an ACTIVE channel.

    This view should be called with timestamp of latest message;
    - When user opens (clicks on) a channel.
    - Periodically (eg: setInterval for 15secs) while user staying in a channel.


    .. code-block:: python

        #  request:
            {
            'view':'_zops_last_seen_msg',
            'channel_key': key,
            'key': key,
            'timestamp': datetime,
            }

        #  response:
            {
            'status': 'OK',
            'code': 200,
            }
    """
    sbs = Subscriber(current).objects.filter(channel_id=current.input['channel_key'],
                                             user_id=current.user_id)[0]
    sbs.last_seen_msg_time = current.input['timestamp']
    sbs.save()
    current.output = {
        'status': 'OK',
        'code': 200}