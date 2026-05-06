def find_l75(contig_lengths_dict, genome_length_dict):
    """
    Calculate the L50 for each strain. L75 is defined as the number of contigs required to achieve the N75
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :param genome_length_dict: dictionary of strain name: total genome length
    :return: l50_dict: dictionary of strain name: L75
    """
    # Initialise the dictionary
    l75_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        currentlength = 0
        currentcontig = 0
        for contig_length in contig_lengths:
            currentlength += contig_length
            currentcontig += 1
            # Same logic as with the L75, but the contig number is added instead of the length of the contig
            if currentlength >= genome_length_dict[file_name] * 0.75:
                l75_dict[file_name] = currentcontig
                break
    return l75_dict