def ensureUniqueIDs(items):
    """
    If action groups are repeated, then links in the table of contents will
    just go to the first of the repeats. This may not be desirable, particularly
    in the case of subcommands where the option groups have different members.
    This function updates the title IDs by adding _repeatX, where X is a number
    so that the links are then unique.
    """
    s = set()
    for item in items:
        for n in item.traverse(descend=True, siblings=True, ascend=False):
            if isinstance(n, nodes.section):
                ids = n['ids']
                for idx, id in enumerate(ids):
                    if id not in s:
                        s.add(id)
                    else:
                        i = 1
                        while "{}_repeat{}".format(id, i) in s:
                            i += 1
                        ids[idx] = "{}_repeat{}".format(id, i)
                        s.add(ids[idx])
                n['ids'] = ids