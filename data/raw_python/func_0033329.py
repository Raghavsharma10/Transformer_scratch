def sumaclust_denovo_cluster(seq_path=None,
                             result_path=None,
                             shortest_len=True,
                             similarity=0.97,
                             threads=1,
                             exact=False,
                             HALT_EXEC=False
                             ):
    """ Function  : launch SumaClust de novo OTU picker

        Parameters: seq_path, filepath to reads;
                    result_path, filepath to output OTU map;
                    shortest_len, boolean;
                    similarity, the similarity threshold (between (0,1]);
                    threads, number of threads to use;
                    exact, boolean to perform exact matching

        Return    : clusters, list of lists
    """

    # Sequence path is mandatory
    if (seq_path is None
            or not exists(seq_path)):
        raise ValueError("Error: FASTA query sequence filepath is "
                         "mandatory input.")

    # Output directory is mandatory
    if (result_path is None
            or not isdir(dirname(realpath(result_path)))):
        raise ValueError("Error: output directory is mandatory input.")

    # Instantiate the object
    sumaclust = Sumaclust(HALT_EXEC=HALT_EXEC)

    # Set the OTU-map filepath
    sumaclust.Parameters['-O'].on(result_path)

    # Set the similarity threshold
    if similarity is not None:
        sumaclust.Parameters['-t'].on(similarity)

    # Set the option to perform exact clustering (default: False)
    if exact:
        sumaclust.Parameters['-e'].on()

    # Turn off option for reference sequence length to be the shortest
    if not shortest_len:
        sumaclust.Parameters['-l'].off()

    # Set the number of threads
    if threads > 0:
        sumaclust.Parameters['-p'].on(threads)
    else:
        raise ValueError("Number of threads must be positive.")

    # Launch SumaClust,
    # set the data string to include the read filepath
    # (to be passed as final arguments in the sumaclust command)
    app_result = sumaclust(seq_path)

    # Put clusters into a list of lists
    f_otumap = app_result['OtuMap']
    clusters = [line.strip().split('\t')[1:] for line in f_otumap]

    # Return clusters
    return clusters