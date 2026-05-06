def find_n75(contig_lengths_dict, genome_length_dict):
    """
    Calculate the N75 for each strain. N75 is defined as the largest contig such that at least 3/4 of the total
    genome size is contained in contigs equal to or larger than this contig
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :param genome_length_dict: dictionary of strain name: total genome length
    :return: n75_dict: dictionary of strain name: N75
    """
    # Initialise the dictionary
    n75_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        currentlength = 0
        for contig_length in contig_lengths:
            currentlength += contig_length
            # If the current length is now greater than the 3/4 of the total genome length, the current contig length
            # is the N75
            if currentlength >= genome_length_dict[file_name] * 0.75:
                n75_dict[file_name] = contig_length
                break
    return n75_dict