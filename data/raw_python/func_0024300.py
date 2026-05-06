def edit_message(current):
    """
    Edit a message a user own.

    .. code-block:: python

        # request:
        {
            'view':'_zops_edit_message',
            'message': {
                'body': string,     # message text
                'key': key
                }
        }
        # response:
            {
            'status': string,   # 'OK' for success
            'code': int,        # 200 for success
            }

    """
    current.output = {'status': 'OK', 'code': 200}
    in_msg = current.input['message']
    try:
        msg = Message(current).objects.get(sender_id=current.user_id, key=in_msg['key'])
        msg.body = in_msg['body']
        msg.save()
    except ObjectDoesNotExist:
        raise HTTPError(404, "")