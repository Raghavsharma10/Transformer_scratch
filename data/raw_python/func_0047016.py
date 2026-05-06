def filter_ribo_counts(counts, orf_start=None, orf_stop=None):
    """Filter read counts  and return only upstream of orf_start or downstream
    of orf_stop.

    Keyword arguments:
    counts -- Ribo-Seq read counts obtained from get_ribo_counts.
    orf_start -- Start position of the longest ORF.
    orf_stop -- Stop position of the longest ORF.

    """
    filtered_counts = dict.copy(counts)
    for position in counts:
        if orf_start and orf_stop:
            # if only upstream and downstream reads are required, check if
            # current position is upstream or downstream of the ORF start/stop
            # if not, remove from counts
            if (position > orf_start and position < orf_stop):
                filtered_counts.pop(position)
        elif orf_start:
            # check if current position is upstream of ORF start. if not, remove
            if position >= orf_start:
                filtered_counts.pop(position)
        elif orf_stop:
            # check if current position is downstream of ORF stop. If not,
            # remove
            if position <= orf_stop:
                filtered_counts.pop(position)

    # calculate total reads for this transcript
    total_reads = sum(sum(item.values()) for item in filtered_counts.values())
    return filtered_counts, total_reads