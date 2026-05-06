def get_clusters_from_fasta_filepath(
        fasta_filepath,
        original_fasta_path,
        percent_ID=0.97,
        max_accepts=1,
        max_rejects=8,
        stepwords=8,
        word_length=8,
        optimal=False,
        exact=False,
        suppress_sort=False,
        output_dir=None,
        enable_rev_strand_matching=False,
        subject_fasta_filepath=None,
        suppress_new_clusters=False,
        return_cluster_maps=False,
        stable_sort=False,
        tmp_dir=gettempdir(),
        save_uc_files=True,
        HALT_EXEC=False):
    """ Main convenience wrapper for using uclust to generate cluster files

    A source fasta file is required for the fasta_filepath.  This will be
    sorted to be in order of longest to shortest length sequences.  Following
    this, the sorted fasta file is used to generate a cluster file in the
    uclust (.uc) format.  Next the .uc file is converted to cd-hit format
    (.clstr).  Finally this file is parsed and returned as a list of lists,
    where each sublist a cluster of sequences.  If an output_dir is
    specified, the intermediate files will be preserved, otherwise all
    files created are temporary and will be deleted at the end of this
    function

    The percent_ID parameter specifies the percent identity for a clusters,
    i.e., if 99% were the parameter, all sequences that were 99% identical
    would be grouped as a cluster.
    """

    # Create readable intermediate filenames if they are to be kept
    fasta_output_filepath = None
    uc_output_filepath = None
    cd_hit_filepath = None

    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'

    if save_uc_files:
        uc_save_filepath = get_output_filepaths(
            output_dir,
            original_fasta_path)
    else:
        uc_save_filepath = None

    sorted_fasta_filepath = ""
    uc_filepath = ""
    clstr_filepath = ""

    # Error check in case any app controller fails
    files_to_remove = []
    try:
        if not suppress_sort:
            # Sort fasta input file from largest to smallest sequence
            sort_fasta = uclust_fasta_sort_from_filepath(fasta_filepath,
                                                         output_filepath=fasta_output_filepath)

            # Get sorted fasta name from application wrapper
            sorted_fasta_filepath = sort_fasta['Output'].name
            files_to_remove.append(sorted_fasta_filepath)

        else:
            sort_fasta = None
            sorted_fasta_filepath = fasta_filepath

        # Generate uclust cluster file (.uc format)
        uclust_cluster = uclust_cluster_from_sorted_fasta_filepath(
            sorted_fasta_filepath,
            uc_save_filepath,
            percent_ID=percent_ID,
            max_accepts=max_accepts,
            max_rejects=max_rejects,
            stepwords=stepwords,
            word_length=word_length,
            optimal=optimal,
            exact=exact,
            suppress_sort=suppress_sort,
            enable_rev_strand_matching=enable_rev_strand_matching,
            subject_fasta_filepath=subject_fasta_filepath,
            suppress_new_clusters=suppress_new_clusters,
            stable_sort=stable_sort,
            tmp_dir=tmp_dir,
            HALT_EXEC=HALT_EXEC)
        # Get cluster file name from application wrapper
        remove_files(files_to_remove)
    except ApplicationError:
        remove_files(files_to_remove)
        raise ApplicationError('Error running uclust. Possible causes are '
                               'unsupported version (current supported version is v1.2.22) is installed or '
                               'improperly formatted input file was provided')
    except ApplicationNotFoundError:
        remove_files(files_to_remove)
        raise ApplicationNotFoundError('uclust not found, is it properly ' +
                                       'installed?')

    # Get list of lists for each cluster
    clusters, failures, seeds = \
        clusters_from_uc_file(uclust_cluster['ClusterFile'])

    # Remove temp files unless user specifies output filepath
    if not save_uc_files:
        uclust_cluster.cleanUp()

    if return_cluster_maps:
        return clusters, failures, seeds
    else:
        return clusters.values(), failures, seeds