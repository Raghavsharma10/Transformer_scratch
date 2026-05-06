def find_contig_distribution(contig_lengths_dict):
    """
    Determine the frequency of different contig size ranges for each strain
    :param contig_lengths_dict:
    :return: contig_len_dist_dict: dictionary of strain name: tuple of contig size range frequencies
    """
    # Initialise the dictionary
    contig_len_dist_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # Initialise integers to store the number of contigs that fall into the different bin sizes
        over_1000000 = 0
        over_500000 = 0
        over_100000 = 0
        over_50000 = 0
        over_10000 = 0
        over_5000 = 0
        other = 0
        for contig_length in contig_lengths:
            # Depending on the size of the contig, increment the appropriate integer
            if contig_length > 1000000:
                over_1000000 += 1
            elif contig_length > 500000:
                over_500000 += 1
            elif contig_length > 100000:
                over_100000 += 1
            elif contig_length > 50000:
                over_50000 += 1
            elif contig_length > 10000:
                over_10000 += 1
            elif contig_length > 5000:
                over_5000 += 1
            else:
                other += 1
        # Populate the dictionary with a tuple of each of the size range frequencies
        contig_len_dist_dict[file_name] = (over_1000000,
                                           over_500000,
                                           over_100000,
                                           over_50000,
                                           over_10000,
                                           over_5000,
                                           other)
    return contig_len_dist_dict