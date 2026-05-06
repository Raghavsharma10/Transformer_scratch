def process_uclust_pw_alignment_results(fasta_pairs_lines, uc_lines):
    """ Process results of uclust search and align """
    alignments = get_next_two_fasta_records(fasta_pairs_lines)
    for hit in get_next_record_type(uc_lines, 'H'):
        matching_strand = hit[4]
        if matching_strand == '-':
            strand_id = '-'
            target_rev_match = True
        elif matching_strand == '+':
            strand_id = '+'
            target_rev_match = False
        elif matching_strand == '.':
            # protein sequence, so no strand information
            strand_id = ''
            target_rev_match = False
        else:
            raise UclustParseError("Unknown strand type: %s" % matching_strand)
        uc_query_id = hit[8]
        uc_target_id = hit[9]
        percent_id = float(hit[3])

        fasta_pair = alignments.next()

        fasta_query_id = fasta_pair[0][0]
        aligned_query = fasta_pair[0][1]

        if fasta_query_id != uc_query_id:
            raise UclustParseError("Order of fasta and uc files do not match." +
                                   " Got query %s but expected %s." %
                                   (fasta_query_id, uc_query_id))

        fasta_target_id = fasta_pair[1][0]
        aligned_target = fasta_pair[1][1]

        if fasta_target_id != uc_target_id + strand_id:
            raise UclustParseError("Order of fasta and uc files do not match." +
                                   " Got target %s but expected %s." %
                                   (fasta_target_id, uc_target_id + strand_id))

        if target_rev_match:
            query_id = uc_query_id + ' RC'
            aligned_query = DNA.rc(aligned_query)
            target_id = uc_target_id
            aligned_target = DNA.rc(aligned_target)
        else:
            query_id = uc_query_id
            aligned_query = aligned_query
            target_id = uc_target_id
            aligned_target = aligned_target

        yield (query_id, target_id, aligned_query, aligned_target, percent_id)