def download_list(user=None, pwd=None, limit=20, offset=0):
    """
    Lists the downloads created by a user.

    :param user: [str] A user name, look at env var ``GBIF_USER`` first
    :param pwd: [str] Your password, look at env var ``GBIF_PWD`` first
    :param limit: [int] Number of records to return. Default: ``20``
    :param offset: [int] Record number to start at. Default: ``0``

    Usage::

      from pygbif import occurrences as occ
      occ.download_list(user = "sckott")
      occ.download_list(user = "sckott", limit = 5)
      occ.download_list(user = "sckott", offset = 21)
    """

    user = _check_environ('GBIF_USER', user)
    pwd = _check_environ('GBIF_PWD', pwd)

    url = 'http://api.gbif.org/v1/occurrence/download/user/' + user
    args = {'limit': limit, 'offset': offset}
    res = gbif_GET(url, args, auth=(user, pwd))
    return {'meta': {'offset': res['offset'],
                     'limit': res['limit'],
                     'endofrecords': res['endOfRecords'],
                     'count': res['count']},
            'results': res['results']}