def edit_channel(current):
    """
        Update channel name or description

        .. code-block:: python

            #  request:
                {
                'view':'_zops_edit_channel,
                'channel_key': key,
                'name': string,
                'description': string,
                }

            #  response:
                {
                'status': 'OK',
                'code': 200
                }
    """
    ch = Channel(current).objects.get(owner_id=current.user_id,
                                      key=current.input['channel_key'])
    ch.name = current.input['name']
    ch.description = current.input['description']
    ch.save()
    for sbs in ch.subscriber_set.objects.all():
        sbs.name = ch.name
        sbs.save()
    current.output = {'status': 'OK', 'code': 200}