def hit_groups(hits):
    """
    * each sequence may have more than one 16S rRNA gene
    * group hits for each gene
    """
    groups = []
    current = False
    for hit in sorted(hits, key = itemgetter(0)):
        if current is False:
            current = [hit]
        elif check_overlap(current, hit) is True or check_order(current, hit) is False:
            groups.append(current)
            current = [hit]
        else:
            current.append(hit)
    groups.append(current)
    return groups