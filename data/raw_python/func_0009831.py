def logs_map_and_reduce(logs, _map, _reduce):
    """
    :type logs str[]
    :type _map (list) -> str
    :type _reduce (list) -> obj
    """
    keys = []
    mapped_count = Counter()
    mapped = defaultdict(list)

    # first map all entries
    for log in logs:
        key = _map(log)
        mapped[key].append(log)
        mapped_count[key] += 1

        if key not in keys:
            keys.append(key)

    # the most common mapped item
    top_count = mapped_count.most_common(1).pop()[1]

    # now reduce mapped items
    reduced = []

    # keep the order under control
    for key in keys:
        entries = mapped[key]
        # print(key, entries)

        # add "value" field to each reduced item (1.0 will be assigned to the most "common" item)
        item = _reduce(entries)
        item['value'] = 1. * len(entries) / top_count

        reduced.append(item)

    # print(mapped)
    return reduced