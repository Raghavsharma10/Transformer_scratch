def container_similarity_vector(container1=None,packages_set=None,by=None):
    '''container similarity_vector is similar to compare_packages, but intended
    to compare a container object (singularity image or singularity hub container)
    to a list of packages. If packages_set is not provided, the default used is 
    'docker-os'. This can be changed to 'docker-library', or if the user wants a custom
    list, should define custom_set.
    :param container1: singularity image or singularity hub container.
    :param packages_set: a name of a package set, provided are docker-os and docker-library
    :by: metrics to compare by (files.txt and or folders.txt)
    '''

    if by == None:
        by = ['files.txt']

    if not isinstance(by,list):
        by = [by]
    if not isinstance(packages_set,list):
        packages_set = [packages_set]

    comparisons = dict()

    for b in by:
        bot.debug("Starting comparisons for %s" %b)
        df = pandas.DataFrame(columns=packages_set)
        for package2 in packages_set:
            sim = calculate_similarity(container1=container1,
                                       image_package2=package2,
                                       by=b)[b]
           
            name1 = os.path.basename(package2).replace('.img.zip','')
            bot.debug("container vs. %s: %s" %(name1,sim))
            df.loc["container",package2] = sim
        df.columns = [os.path.basename(x).replace('.img.zip','') for x in df.columns.tolist()]
        comparisons[b] = df
    return comparisons