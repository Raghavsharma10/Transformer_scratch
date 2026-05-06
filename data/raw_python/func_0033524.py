def parse_cdhit_clstr_file(lines):
    """Returns a list of list of sequence ids representing clusters"""
    clusters = []
    curr_cluster = []

    for l in lines:
        if l.startswith('>Cluster'):
            if not curr_cluster:
                continue
            clusters.append(curr_cluster)
            curr_cluster = []
        else:
            curr_cluster.append(clean_cluster_seq_id(l.split()[2]))

    if curr_cluster:
        clusters.append(curr_cluster)

    return clusters