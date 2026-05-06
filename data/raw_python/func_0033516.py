def parse_dereplicated_uc(dereplicated_uc_lines):
    """ Return dict of seq ID:dereplicated seq IDs from dereplicated .uc lines

    dereplicated_uc_lines: list of lines of .uc file from dereplicated seqs from
     usearch61 (i.e. open file of abundance sorted .uc data)
    """

    dereplicated_clusters = {}

    seed_hit_ix = 0
    seq_id_ix = 8
    seed_id_ix = 9

    for line in dereplicated_uc_lines:
        if line.startswith("#") or len(line.strip()) == 0:
            continue
        curr_line = line.strip().split('\t')
        if curr_line[seed_hit_ix] == "S":
            dereplicated_clusters[curr_line[seq_id_ix]] = []
        if curr_line[seed_hit_ix] == "H":
            curr_seq_id = curr_line[seq_id_ix]
            dereplicated_clusters[curr_line[seed_id_ix]].append(curr_seq_id)

    return dereplicated_clusters