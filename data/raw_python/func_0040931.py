def find_genome_length(contig_lengths_dict):
    """
    Determine the total length of all the contigs for each strain
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :return: genome_length_dict: dictionary of strain name: total genome length
    """
    # Initialise the dictionary
    genome_length_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # Use the sum() method to add all the contig lengths in the list
        genome_length_dict[file_name] = sum(contig_lengths)
    return genome_length_dict