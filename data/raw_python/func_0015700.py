def signal_list_names(type_):
    """Returns a list of signal names for the given type

    :param type\\_:
    :type type\\_: :obj:`GObject.GType`
    :returns: A list of signal names
    :rtype: :obj:`list`
    """

    ids = signal_list_ids(type_)
    return tuple(GObjectModule.signal_name(i) for i in ids)