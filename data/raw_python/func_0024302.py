def unflag_message(current):
    """
    remove flag of a message

    .. code-block:: python

        # request:
        {
            'view':'_zops_flag_message',
            'key': key,
        }
        # response:
            {
            '
            'status': 'OK',
            'code': 200,
            }

    """
    current.output = {'status': 'OK', 'code': 200}

    FlaggedMessage(current).objects.filter(user_id=current.user_id,
                                           message_id=current.input['key']).delete()