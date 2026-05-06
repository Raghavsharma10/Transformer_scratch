def exchange_oauth2_member(access_token, base_url=OH_BASE_URL,
                           all_files=False):
    """
    Returns data for a specific user, including shared data files.

    :param access_token: This field is the user specific access_token.
    :param base_url: It is this URL `https://www.openhumans.org`.
    """
    url = urlparse.urljoin(
        base_url,
        '/api/direct-sharing/project/exchange-member/?{}'.format(
            urlparse.urlencode({'access_token': access_token})))
    member_data = get_page(url)

    returned = member_data.copy()

    # Get all file data if all_files is True.
    if all_files:
        while member_data['next']:
            member_data = get_page(member_data['next'])
            returned['data'] = returned['data'] + member_data['data']

    logging.debug('JSON data: {}'.format(returned))
    return returned