def get_notifications(current):
    """
        Returns last N notifications for current user


        .. code-block:: python

            #  request:
                {
                'view':'_zops_unread_messages',
                'amount': int, # Optional, defaults to 8
                }

            #  response:
                {
                'status': 'OK',
                'code': 200,
                'notifications': [{'title':string,
                                   'body': string,
                                   'channel_key': key,
                                   'type': int,
                                   'url': string, # could be a in app JS URL prefixed with "#" or
                                                  # full blown URL prefixed with "http"
                                   'message_key': key,
                                   'timestamp': datetime},],
                }
        """
    current.output = {
        'status': 'OK',
        'code': 200,
        'notifications': [],
    }
    amount = current.input.get('amount', 8)
    try:
        notif_sbs = current.user.subscriptions.objects.get(channel_id=current.user.prv_exchange)
    except MultipleObjectsReturned:
        # FIXME: This should not happen,
        log.exception("MULTIPLE PRV EXCHANGES!!!!")
        sbs = current.user.subscriptions.objects.filter(channel_id=current.user.prv_exchange)
        sbs[0].delete()
        notif_sbs = sbs[1]
    for msg in notif_sbs.channel.message_set.objects.all()[:amount]:
        current.output['notifications'].insert(0, {
            'title': msg.msg_title,
            'body': msg.body,
            'type': msg.typ,
            'url': msg.url,
            'channel_key': msg.channel.key,
            'message_key': msg.key,
            'timestamp': msg.updated_at})