def filter_data_columns(data):
    """
    Given a dict of data such as those in :py:class:`~.ProjectStats` attributes,
    made up of :py:class:`datetime.datetime` keys and values of dicts of column
    keys to counts, return a list of the distinct column keys in sorted order.

    :param data: data dict as returned by ProjectStats attributes
    :type data: dict
    :return: sorted list of distinct keys
    :rtype: ``list``
    """
    keys = set()
    for dt, d in data.items():
        for k in d:
            keys.add(k)
    return sorted([x for x in keys])