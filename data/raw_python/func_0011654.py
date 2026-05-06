def _check_title_length(title, api):
    ''' Checks the given title against the given API specifications to ensure it's short enough '''
    if len(title) > api.MAX_SUBTASK_TITLE_LENGTH:
        raise ValueError("Title cannot be longer than {} characters".format(api.MAX_SUBTASK_TITLE_LENGTH))