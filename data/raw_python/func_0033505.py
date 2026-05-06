def enumerate_otus(fasta_filepath,
                   output_filepath=None,
                   label_prefix="",
                   label_suffix="",
                   retain_label_as_comment=False,
                   count_start=0):
    """ Writes unique, sequential count to OTUs

    fasta_filepath = input fasta filepath
    output_filepath = output fasta filepath
    label_prefix = string to place before enumeration
    label_suffix = string to place after enumeration
    retain_label_as_comment = if True, will place existing label in sequence
     comment, after a tab
    count_start = number to start enumerating OTUs with

    """

    fasta_i = open(fasta_filepath, "U")

    if not output_filepath:
        _, output_filepath = mkstemp(prefix='enumerated_seqs_',
                                     suffix='.fasta')

    fasta_o = open(output_filepath, "w")

    for label, seq in parse_fasta(fasta_i):
        curr_label = ">" + label_prefix + str(count_start) + label_suffix
        if retain_label_as_comment:
            curr_label += '\t' + label
        fasta_o.write(curr_label.strip() + '\n')
        fasta_o.write(seq.strip() + '\n')
        count_start += 1

    return output_filepath