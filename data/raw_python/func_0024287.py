def unread_count(current):
    """
        Number of unread messages for current user


        .. code-block:: python

            #  request:
                {
                'view':'_zops_unread_count',
                }

            #  response:
                {
                'status': 'OK',
                'code': 200,
                'notifications': int,
                'messages': int,
                }
        """
    unread_ntf = 0
    unread_msg = 0
    for sbs in current.user.subscriptions.objects.filter(is_visible=True):
        try:
            if sbs.channel.key == current.user.prv_exchange:
                unread_ntf += sbs.unread_count()
            else:
                unread_msg += sbs.unread_count()
        except ObjectDoesNotExist:
            # FIXME: This should not happen,
            log.exception("MULTIPLE PRV EXCHANGES!!!!")
            sbs.delete()
    current.output = {
        'status': 'OK',
        'code': 200,
        'notifications': unread_ntf,
        'messages': unread_msg
    }