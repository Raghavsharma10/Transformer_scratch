def fasta_stats(files, records):
    """
    Parse the lengths of all contigs for each sample, as well as the total GC%
    :param files: dictionary of stain name: /sequencepath/strain_name.extension
    :param records: Dictionary of strain name: SeqIO records
    :return: contig_len_dict, gc_dict: dictionaries of list of all contig length, and total GC% for all strains
    """
    # Initialise dictionaries
    contig_len_dict = dict()
    gc_dict = dict()
    for file_name in files:
        # Initialise variables to store appropriate values parsed from contig records
        contig_lengths = list()
        fasta_sequence = str()
        for contig, record in records[file_name].items():
            # Append the length of the contig to the list
            contig_lengths.append(len(record.seq))
            # Add the contig sequence to the string
            fasta_sequence += record.seq
        # Set the reverse sorted (e.g. largest to smallest) list of contig sizes as the value
        contig_len_dict[file_name] = sorted(contig_lengths, reverse=True)
        # Calculate the GC% of the total genome sequence using GC - format to have two decimal places
        gc_dict[file_name] = float('{:0.2f}'.format(GC(fasta_sequence)))
    return contig_len_dict, gc_dict