def parse(filename, MAX_TERM_COUNT=1000):
    """
    MAX_TERM_COUNT = 10000       # There are 39,000 terms in the GO!
    """
    with open(filename, "r") as f:

        termId = None
        name = None
        desc = None
        parents = []

        termCount = 0
        for l in f.readlines():
            if l.startswith("id:"):
                termId = l.strip()[4:]
            if l.startswith("name:"):
                name = l.strip()[6:]
            elif l.startswith("def:"):
                desc = l.strip()[5:]
            elif l.startswith("is_a:"):
                pid = l.strip()[6:].split(" ", 1)[0]
                parents.append(pid)
            if len(l) == 1:     # newline
                # save
                if termId is not None and name is not None:
                    terms[termId] = {'name': name, 'desc': desc,
                                     'parents': parents[:], 'children': []}
                    termId = None
                    name = None
                    parents = []
                    termCount += 1
                    if MAX_TERM_COUNT is not None and \
                       termCount > MAX_TERM_COUNT:
                        break

    count = 0
    for tid, tdict in terms.items():
        count += 1      # purely for display
        for p in tdict['parents']:
            if p in terms.keys():
                terms[p]['children'].append(tid)

    # Get unique term IDs for Tag Groups.
    tagGroups = set()
    for tid, tdict in terms.items():
        # Only create Tags for GO:terms that are 'leafs' of the tree
        if len(tdict['children']) == 0:
            for p in tdict['parents']:
                tagGroups.add(p)

    return tagGroups, terms