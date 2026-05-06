def compare_containers(container1=None, container2=None):

    '''compare_containers will generate a data structure with common and unique files to
    two images. If environmental variable SINGULARITY_HUB is set, will use container
    database objects.
    :param container1: first container for comparison
    :param container2: second container for comparison if either not defined must include
    default compares just files
    '''

    # Get files and folders for each
    container1_guts = get_container_contents(split_delim="\n",
                                             container=container1)['all']
    container2_guts = get_container_contents(split_delim="\n",
                                             container=container2)['all']

    # Do the comparison for each metric
    return compare_lists(container1_guts, container2_guts)