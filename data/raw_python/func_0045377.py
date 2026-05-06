def sort_variant_file(infile):
    """
    Sort a modified variant file.
    Sorting is based on the first column and the POS.
    
    Uses unix sort to sort the variants and overwrites the infile.
    
    Args:
        infile : A string that is the path to a file
        mode : 'chromosome' or 'rank'
        outfile : The path to an outfile where the variants should be printed
    
    Returns:
        0 if sorting was performed
        1 if variants where not sorted
    """
    command = [
            'sort',
            ]
    command.append('-n')
    command.append('-k1')
    command.append('-k3')

    command = command + [infile, '-o', infile]

    logger.info("Start sorting variants...")
    logger.info("Sort command: {0}".format(' '.join(command)))
    sort_start = datetime.now()
    
    try:
        call(command)
    except OSError as e:
        logger.warning("unix command sort does not seem to exist on your system...")
        logger.warning("genmod needs unix sort to provide a sorted output.")
        logger.warning("Output VCF will not be sorted since genmod can not find"\
                        "unix sort")
        raise e

    logger.info("Sorting done. Time to sort: {0}".format(datetime.now()-sort_start))
    
    return