def flag_message(current):
    """
    Flag inappropriate messages

    .. code-block:: python

        # request:
        {
            'view':'_zops_flag_message',
            'message_key': key,
        }
        # response:
            {
            '
            'status': 'Created',
            'code': 201,
            }

    """
    current.output = {'status': 'Created', 'code': 201}
    FlaggedMessage.objects.get_or_create(user_id=current.user_id,
                                         message_id=current.input['key'])