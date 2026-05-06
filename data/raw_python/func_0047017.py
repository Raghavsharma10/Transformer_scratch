def get_ribo_counts(ribo_fileobj, transcript_name, read_lengths, read_offsets):
    """For each mapped read of the given transcript in the BAM file
    (pysam AlignmentFile object), return the position (+1) and the
    corresponding frame (1, 2 or 3) to which it aligns.

    Keyword arguments:
    ribo_fileobj -- file object - BAM file opened using pysam AlignmentFile
    transcript_name -- Name of transcript to get counts for
    read_length (optional) -- If provided, get counts only for reads of this length.

    """
    read_counts = {}
    total_reads = 0
    for record in ribo_fileobj.fetch(transcript_name):
        query_length = record.query_length
        position_ref = record.pos + 1
        for index, read_length in enumerate(read_lengths):
            position = position_ref  # reset position
            if read_length == 0 or read_length == query_length:
                # if an offset is specified, increment position by that offset.
                position += read_offsets[index]
            else:
                # ignore other reads/lengths
                continue
            total_reads += 1
            try:
                read_counts[position]
            except KeyError:
                read_counts[position] = {1: 0, 2: 0, 3: 0}

            # calculate the frame of the read from position
            rem = position % 3
            if rem == 0:
                read_counts[position][3] += 1
            else:
                read_counts[position][rem] += 1

    log.debug('Total read counts: {}'.format(total_reads))
    log.debug('RiboSeq read counts for transcript: {0}\n{1}'.format(transcript_name, read_counts))
    return read_counts, total_reads