def sgraph(N_clusters_max, file_name):
    """Runs METIS or hMETIS and returns the labels found by those 
        (hyper-)graph partitioning algorithms.
        
    Parameters
    ----------
    N_clusters_max : int
    
    file_name : string
    
    Returns
    -------
    labels : array of shape (n_samples,)
        A vector of labels denoting the cluster to which each sample has been assigned
        as a result of any of three approximation algorithms for consensus clustering 
        (either of CSPA, HGPA or MCLA).
    """

    if file_name == 'DO_NOT_PROCESS':
        return []

    print('\n#')

    k = str(N_clusters_max)
    out_name = file_name + '.part.' + k
    if file_name == 'wgraph_HGPA':
        print("INFO: Cluster_Ensembles: sgraph: "
              "calling shmetis for hypergraph partitioning.")
        
        if sys.platform.startswith('linux'):
            shmetis_path = pkg_resources.resource_filename(__name__, 
                                         'Hypergraph_Partitioning/hmetis-1.5-linux/shmetis')
        elif sys.platform.startswith('darwin'):
            shmetis_path = pkg_resources.resource_filename(__name__, 
                                      'Hypergraph_Partitioning/hmetis-1.5-osx-i686/shmetis')
        else:
            print("ERROR: Cluster_Ensembles: sgraph:\n"
                  "your platform is not supported. Some code required for graph partition "
                  "is only available for Linux distributions and OS X.")
            sys.exit(1)
        
        args = "{0} ./".format(shmetis_path) + file_name + " " + k + " 15"
        subprocess.call(args, shell = True)
    elif file_name == 'wgraph_CSPA' or file_name == 'wgraph_MCLA':
        print("INFO: Cluster_Ensembles: sgraph: "
              "calling gpmetis for graph partitioning.")
        args = "gpmetis ./" + file_name + " " + k
        subprocess.call(args, shell = True)
    else:
        raise NameError("ERROR: Cluster_Ensembles: sgraph: {} is not an acceptable "
                        "file-name.".format(file_name))

    labels = np.empty(0, dtype = int)
    with open(out_name, 'r') as file:
        print("INFO: Cluster_Ensembles: sgraph: (hyper)-graph partitioning completed; "
              "loading {}".format(out_name))
        labels = np.loadtxt(out_name, dtype = int)
        labels = labels.reshape(labels.size)
    labels = one_to_max(labels)            

    subprocess.call(['rm', out_name])

    print('#')

    return labels