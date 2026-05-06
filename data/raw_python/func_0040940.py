def find_orf_distribution(orf_file_dict):
    """
    Parse the prodigal outputs to determine the frequency of ORF size ranges for each strain
    :param orf_file_dict: dictionary of strain name: /sequencepath/prodigal results.sco
    :return: orf_dist_dict: dictionary of strain name: tuple of ORF size range distribution frequencies
    """
    # Initialise the dictionary
    orf_dist_dict = dict()
    for file_name, orf_report in orf_file_dict.items():
        # Initialise variable to store the frequency of the different ORF size ranges
        total_orfs = 0
        over_3000 = 0
        over_1000 = 0
        over_500 = 0
        other = 0
        # Open the strain-specific report
        with open(orf_report, 'r') as orfreport:
            for line in orfreport:
                # The report has a header section that can be ignored - only parse lines beginning with '>'
                if line.startswith('>'):
                    # Split the line on '_' characters e.g. >1_345_920_- yields contig: >1, start: 345, stop: 920,
                    # direction: -
                    contig, start, stop, direction = line.split('_')
                    # The size of the ORF is the end position minus the start position e.g. 920 - 345 = 575
                    size = int(stop) - int(start)
                    # Increment the total number of ORFs before binning based on ORF size
                    total_orfs += 1
                    # Increment the appropriate integer based on ORF size
                    if size > 3000:
                        over_3000 += 1
                    elif size > 1000:
                        over_1000 += 1
                    elif size > 500:
                        over_500 += 1
                    else:
                        other += 1
        # Populate the dictionary with a tuple of the ORF size range frequencies
        orf_dist_dict[file_name] = (total_orfs,
                                    over_3000,
                                    over_1000,
                                    over_500,
                                    other)
        # Clean-up the prodigal reports
        try:
            os.remove(orf_report)
        except IOError:
            pass
    return orf_dist_dict