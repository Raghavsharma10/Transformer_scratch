def find_num_contigs(contig_lengths_dict):
    """
    Count the total number of contigs for each strain
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :return: num_contigs_dict: dictionary of strain name: total number of contigs
    """
    # Initialise the dictionary
    num_contigs_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # Use the len() method to count the number of entries in the list
        num_contigs_dict[file_name] = len(contig_lengths)
    return num_contigs_dict