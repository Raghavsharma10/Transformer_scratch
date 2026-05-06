def message(subject, message, access_token, all_members=False,
            project_member_ids=None, base_url=OH_BASE_URL):
    """
    Send an email to individual users or in bulk. To learn more about Open
    Humans OAuth2 projects, go to:
    https://www.openhumans.org/direct-sharing/oauth2-features/

    :param subject: This field is the subject of the email.
    :param message: This field is the body of the email.
    :param access_token: This is user specific access token/master token.
    :param all_members: This is a boolean field to send email to all members of
        the project.
    :param project_member_ids: This field is the list of project_member_id.
    :param base_url: It is this URL `https://www.openhumans.org`.
    """
    url = urlparse.urljoin(
        base_url, '/api/direct-sharing/project/message/?{}'.format(
            urlparse.urlencode({'access_token': access_token})))
    if not(all_members) and not(project_member_ids):
        response = requests.post(url, data={'subject': subject,
                                            'message': message})
        handle_error(response, 200)
        return response
    elif all_members and project_member_ids:
        raise ValueError(
            "One (and only one) of the following must be specified: "
            "project_members_id or all_members is set to True.")
    else:
        r = requests.post(url, data={'all_members': all_members,
                                     'project_member_ids': project_member_ids,
                                     'subject': subject,
                                     'message': message})
        handle_error(r, 200)
        return r