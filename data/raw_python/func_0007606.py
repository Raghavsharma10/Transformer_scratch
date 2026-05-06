def message_cli(subject, message_body, access_token, all_members=False,
                project_member_ids=None, base_url=OH_BASE_URL,
                verbose=False, debug=False):
    """
    Command line function for sending email to a single user or in bulk.
    For more information visit
    :func:`message<ohapi.api.message>`.

    """
    if project_member_ids:
        project_member_ids = re.split(r'[ ,\r\n]+', project_member_ids)
    return message(subject, message_body, access_token, all_members,
                   project_member_ids, base_url)