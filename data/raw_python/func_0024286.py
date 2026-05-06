def list_channels(current):
    """
        List channel memberships of current user


        .. code-block:: python

            #  request:
                {
                'view':'_zops_list_channels',
                }

            #  response:
                {
                'channels': [
                    {'name': string, # name of channel
                     'key': key,     # key of channel
                     'unread': int,  # unread message count
                     'type': int,    # channel type,
                                     # 15: public channels (chat room/broadcast channel distinction
                                                         comes from "read_only" flag)
                                     # 10: direct channels
                                     # 5: one and only private channel which is "Notifications"
                     'read_only': boolean,
                                     # true if this is a read-only subscription to a broadcast channel
                                     # false if it's a public chat room

                     'actions':[('action name', 'view name'),]
                    },]
                }
        """
    current.output = {
        'status': 'OK',
        'code': 200,
        'channels': []}
    for sbs in current.user.subscriptions.objects.filter(is_visible=True):
        try:
            current.output['channels'].append(sbs.get_channel_listing())
        except ObjectDoesNotExist:
            # FIXME: This should not happen,
            log.exception("UNPAIRED DIRECT EXCHANGES!!!!")
            sbs.delete()