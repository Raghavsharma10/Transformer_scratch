def get_message_actions(current):
    """
    Returns applicable actions for current user for given message key

    .. code-block:: python

        # request:
        {
            'view':'_zops_get_message_actions',
            'key': key,
        }
        # response:
            {
            'actions':[('name_string', 'cmd_string'),]
            'status': string,   # 'OK' for success
            'code': int,        # 200 for success
            }

    """
    current.output = {'status': 'OK',
                      'code': 200,
                      'actions': Message.objects.get(
                          current.input['key']).get_actions_for(current.user)}