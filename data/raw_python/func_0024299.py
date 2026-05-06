def delete_message(current):
    """
        Delete a message

        .. code-block:: python

            #  request:
                {
                'view':'_zops_delete_message,
                'message_key': key,
                }

            #  response:
                {
                'key': key,
                'status': 'OK',
                'code': 200
                }
    """
    try:
        Message(current).objects.get(sender_id=current.user_id,
                                     key=current.input['key']).delete()
        current.output = {'status': 'Deleted', 'code': 200, 'key': current.input['key']}
    except ObjectDoesNotExist:
        raise HTTPError(404, "")