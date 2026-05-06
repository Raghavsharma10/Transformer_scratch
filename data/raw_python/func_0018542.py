def generate(tagGroups, terms):
    """
    create Tag Groups and Child Tags using data from terms dict
    """

    rv = []
    for pid in tagGroups:
        # In testing we may not have complete set
        if pid not in terms.keys():
            continue

        groupData = terms[pid]
        groupName = "[%s] %s" % (pid, groupData['name'])
        groupDesc = groupData['desc']
        children = []
        group = dict(name=groupName, desc=groupDesc, set=children)
        rv.append(group)

        for cid in groupData['children']:
            cData = terms[cid]
            cName = "[%s] %s" % (cid, cData['name'])
            cDesc = cData['desc']
            child = dict(name=cName, desc=cDesc)
            children.append(child)

    return json.dumps(rv, indent=2)