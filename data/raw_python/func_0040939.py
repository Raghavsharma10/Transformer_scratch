def predict_orfs(file_dict, num_threads=1):
    """
    Use prodigal to predict the number of open reading frames (ORFs) in each strain
    :param file_dict: dictionary of strain name: /sequencepath/strain_name.extension
    :param num_threads: number of threads to use in the pool of prodigal processes
    :return: orf_file_dict: dictionary of strain name: /sequencepath/prodigal results.sco
    """
    # Initialise the dictionary
    orf_file_dict = dict()
    prodigallist = list()
    for file_name, file_path in file_dict.items():
        # Set the name of the output .sco results file
        results = os.path.splitext(file_path)[0] + '.sco'
        # Create the command for prodigal to execute - use sco output format
        prodigal = ['prodigal', '-i', file_path, '-o', results,  '-f',  'sco']
        # Only run prodigal if the output file doesn't already exist
        if not os.path.isfile(results):
            prodigallist.append(prodigal)
        # Populate the dictionary with the name of the results file
        orf_file_dict[file_name] = results
    # Setup the multiprocessing pool.
    pool = multiprocessing.Pool(processes=num_threads)
    pool.map(run_prodigal, prodigallist)
    pool.close()
    pool.join()
    return orf_file_dict