def logout(current):
    """
    Log out view.
    Simply deletes the session object.
    For showing logout message:
        'show_logout_message' field should be True in current.task_data,
        Message should be sent in current.task_data with 'logout_message' field.
        Message title should be sent in current.task_data with 'logout_title' field.

        current.task_data['show_logout_message'] = True
        current.task_data['logout_title'] = 'Message Title'
        current.task_data['logout_message'] = 'Message'

    Args:
        current: :attr:`~zengine.engine.WFCurrent` object.
    """
    current.user.is_online(False)
    current.session.delete()
    current.output['cmd'] = 'logout'
    if current.task_data.get('show_logout_message', False):
        current.output['title'] = current.task_data.get('logout_title', None)
        current.output['msg'] = current.task_data.get('logout_message', None)