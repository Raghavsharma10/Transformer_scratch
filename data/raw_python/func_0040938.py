def find_l90(contig_lengths_dict, genome_length_dict):
    """
    Calculate the L90 for each strain. L90 is defined as the number of contigs required to achieve the N90
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :param genome_length_dict: dictionary of strain name: total genome length
    :return: l90_dict: dictionary of strain name: L90
    """
    # Initialise the dictionary
    l90_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        currentlength = 0
        # Initialise a variable to count how many contigs have been added to the currentlength variable
        currentcontig = 0
        for contig_length in contig_lengths:
            currentlength += contig_length
            # Increment :currentcontig each time a contig is added to the current length
            currentcontig += 1
            # Same logic as with the N50, but the contig number is added instead of the length of the contig
            if currentlength >= genome_length_dict[file_name] * 0.9:
                l90_dict[file_name] = currentcontig
                break
    return l90_dict