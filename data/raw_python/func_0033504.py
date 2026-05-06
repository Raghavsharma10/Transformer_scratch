def concatenate_fastas(output_fna_clustered,
                       output_fna_failures,
                       output_concat_filepath):
    """ Concatenates two input fastas, writes to output_concat_filepath

    output_fna_clustered: fasta of successful ref clusters
    output_fna_failures: de novo fasta of cluster failures
    output_concat_filepath: path to write combined fastas to
    """

    output_fp = open(output_concat_filepath, "w")

    for label, seq in parse_fasta(open(output_fna_clustered, "U")):
        output_fp.write(">%s\n%s\n" % (label, seq))
    for label, seq in parse_fasta(open(output_fna_failures, "U")):
        output_fp.write(">%s\n%s\n" % (label, seq))

    return output_concat_filepath