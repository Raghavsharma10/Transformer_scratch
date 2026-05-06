def find_n50(contig_lengths_dict, genome_length_dict):
    """
    Calculate the N50 for each strain. N50 is defined as the largest contig such that at least half of the total
    genome size is contained in contigs equal to or larger than this contig
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :param genome_length_dict: dictionary of strain name: total genome length
    :return: n50_dict: dictionary of strain name: N50
    """
    # Initialise the dictionary
    n50_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # Initialise a variable to store a running total of contig lengths
        currentlength = 0
        for contig_length in contig_lengths:
            # Increment the current length with the length of the current contig
            currentlength += contig_length
            # If the current length is now greater than the total genome / 2, the current contig length is the N50
            if currentlength >= genome_length_dict[file_name] * 0.5:
                # Populate the dictionary, and break the loop
                n50_dict[file_name] = contig_length
                break
    return n50_dict