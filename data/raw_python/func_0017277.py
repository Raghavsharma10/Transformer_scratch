def fuzzy_match_tipnames(ttree, names, wildcard, regex, mrca=True, mono=True):
    """
    Used in multiple internal functions (e.g., .root()) and .drop_tips())
    to select an internal mrca node, or multiple tipnames, using fuzzy matching
    so that every name does not need to be written out by hand.

    name: verbose list
    wildcard: matching unique string
    regex: regex expression
    mrca: return mrca node of selected tipnames. 
    mono: raise error if selected tipnames are not monophyletic    
    """
    # require arguments
    if not any([names, wildcard, regex]):
        raise ToytreeError(
            "must enter an outgroup, wildcard selector, or regex pattern")

    # get list of **nodes** from {list, wildcard, or regex}
    tips = []
    if names:
        if isinstance(names, (str, int)):
            names = [names]
        notfound = [i for i in names if i not in ttree.get_tip_labels()]
        if any(notfound):
            raise ToytreeError(
                "Sample {} is not in the tree".format(notfound))
        tips = [i for i in ttree.treenode.get_leaves() if i.name in names]

    # use regex to match tipnames
    elif regex:
        tips = [
            i for i in ttree.treenode.get_leaves() if re.match(regex, i.name)
        ]               
        if not any(tips):
            raise ToytreeError("No Samples matched the regular expression")

    # use wildcard substring matching
    elif wildcard:
        tips = [i for i in ttree.treenode.get_leaves() if wildcard in i.name]
        if not any(tips):
            raise ToytreeError("No Samples matched the wildcard")

    # build list of **tipnames** from matched nodes
    if not tips:
        raise ToytreeError("no matching tipnames")       
    tipnames = [i.name for i in tips]

    # if a single tipname matched no need to check for monophyly
    if len(tips) == 1:
        if mrca:
            return tips[0]
        else:
            return tipnames

    # if multiple nodes matched, check if they're monophyletic
    mbool, mtype, mnames = (
        ttree.treenode.check_monophyly(
            tipnames, "name", ignore_missing=True)
    )

    # get mrca node
    node = ttree.treenode.get_common_ancestor(tips)

    # raise an error if required to be monophyletic but not
    if mono:
        if not mbool:
            raise ToytreeError(
                "Taxon list cannot be paraphyletic")

    # return tips or nodes
    if not mrca:
        return tipnames
    else:
        return node