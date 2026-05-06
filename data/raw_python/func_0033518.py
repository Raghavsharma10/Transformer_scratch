def merge_clusters_dereplicated_seqs(de_novo_clusters,
                                     dereplicated_clusters):
    """ combines de novo clusters and dereplicated seqs to OTU id:seqs dict

    de_novo_clusters: dict of OTU ID:clustered sequences
    dereplicated_clusters:  dict of seq IDs: dereplicated seq IDs
    """

    clusters = {}

    for curr_denovo_key in de_novo_clusters.keys():
        clusters[curr_denovo_key] = de_novo_clusters[curr_denovo_key]
        curr_clusters = []
        for curr_denovo_id in de_novo_clusters[curr_denovo_key]:
            curr_clusters += dereplicated_clusters[curr_denovo_id]
        clusters[curr_denovo_key] += curr_clusters

    return clusters