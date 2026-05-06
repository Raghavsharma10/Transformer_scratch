def parse_share_url(share_url):
    """Return the group_id and share_token in a group's share url.

    :param str share_url: the share url of a group
    """
    *__, group_id, share_token = share_url.rstrip('/').split('/')
    return group_id, share_token