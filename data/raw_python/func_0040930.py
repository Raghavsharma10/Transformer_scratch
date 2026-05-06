def find_largest_contig(contig_lengths_dict):
    """
    Determine the largest contig for each strain
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :return: longest_contig_dict: dictionary of strain name: longest contig
    """
    # Initialise the dictionary
    longest_contig_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # As the list is sorted in descending order, the largest contig is the first entry in the list
        longest_contig_dict[file_name] = contig_lengths[0]
    return longest_contig_dict