def merge_failures_dereplicated_seqs(failures,
                                     dereplicated_clusters):
    """ Appends failures from dereplicated seqs to failures list

    failures: list of failures
    dereplicated_clusters:  dict of seq IDs: dereplicated seq IDs
    """

    curr_failures = set(failures)
    dereplicated_ids = set(dereplicated_clusters)

    for curr_failure in curr_failures:
        if curr_failure in dereplicated_ids:
            failures += dereplicated_clusters[curr_failure]

    return failures