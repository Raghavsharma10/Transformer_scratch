def create_message(current):
    """
    Creates a message for the given channel.

    .. code-block:: python

        # request:
        {
            'view':'_zops_create_message',
            'message': {
                'channel': key,     # of channel
                'body': string,     # message text.,
                'type': int,        # zengine.messaging.model.MSG_TYPES,
                'attachments': [{
                    'description': string,  # can be blank,
                    'name': string,         # file name with extension,
                    'content': string,      # base64 encoded file content
                    }]}
        # response:
            {
            'status': 'Created',
            'code': 201,
            'msg_key': key,     # key of the message object,
            }

    """
    msg = current.input['message']
    msg_obj = Channel.add_message(msg['channel'], body=msg['body'], typ=msg['type'],
                                  sender=current.user,
                                  title=msg['title'], receiver=msg['receiver'] or None)
    current.output = {
        'msg_key': msg_obj.key,
        'status': 'Created',
        'code': 201
    }
    if 'attachment' in msg:
        for atch in msg['attachments']:
            typ = current._dedect_file_type(atch['name'], atch['content'])
            Attachment(channel_id=msg['channel'], msg=msg_obj, name=atch['name'],
                       file=atch['content'], description=atch['description'], typ=typ).save()