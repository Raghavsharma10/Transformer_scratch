def parse_usearch61_failures(seq_path,
                             failures,
                             output_fasta_fp):
    """ Parses seq IDs from failures list, writes to output_fasta_fp

    seq_path: filepath of original input fasta file.
    failures: list/set of failure seq IDs
    output_fasta_fp: path to write parsed sequences
    """

    parsed_out = open(output_fasta_fp, "w")

    for label, seq in parse_fasta(open(seq_path), "U"):
        curr_label = label.split()[0]
        if curr_label in failures:
            parsed_out.write(">%s\n%s\n" % (label, seq))
    parsed_out.close()
    return output_fasta_fp