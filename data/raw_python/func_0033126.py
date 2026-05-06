def swarm_denovo_cluster(seq_path,
                         d=1,
                         threads=1,
                         HALT_EXEC=False):
    """ Function  : launch the Swarm de novo OTU picker

        Parameters: seq_path, filepath to reads
                    d, resolution
                    threads, number of threads to use

        Return    : clusters, list of lists
    """

    # Check sequence file exists
    if not exists(seq_path):
        raise ValueError("%s does not exist" % seq_path)

    # Instantiate the object
    swarm = Swarm(HALT_EXEC=HALT_EXEC)

    # Set the resolution
    if d > 0:
        swarm.Parameters['-d'].on(d)
    else:
        raise ValueError("Resolution -d must be a positive integer.")

    # Set the number of threads
    if threads > 0:
        swarm.Parameters['-t'].on(threads)
    else:
        raise ValueError("Number of threads must be a positive integer.")

    # create temporary file for Swarm OTU-map
    f, tmp_swarm_otumap = mkstemp(prefix='temp_otumap_',
                                  suffix='.swarm')
    close(f)

    swarm.Parameters['-o'].on(tmp_swarm_otumap)

    # Remove this file later, the final OTU-map
    # is output by swarm_breaker.py and returned
    # as a list of lists (clusters)
    swarm.files_to_remove.append(tmp_swarm_otumap)

    # Launch Swarm
    # set the data string to include the read filepath
    # (to be passed as final arguments in the swarm command)
    clusters = swarm(seq_path)

    remove_files(swarm.files_to_remove, error_on_missing=False)

    # Return clusters
    return clusters