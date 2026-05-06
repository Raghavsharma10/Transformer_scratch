def get_members_by_source(base_url=BASE_URL_API):
    """
    Function returns which members have joined each activity.

    :param base_url: It is URL: `https://www.openhumans.org/api/public-data`.
    """
    url = '{}members-by-source/'.format(base_url)
    response = get_page(url)
    return response