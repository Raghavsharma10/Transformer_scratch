def read_igor_D_gene_parameters(params_file_name):
    """Load genD from file.
    
    genD is a list of genomic D information. Each element is a list of the name
    of the D allele and the germline sequence.

    Parameters
    ----------
    params_file_name : str
        File name for a IGOR parameter file.

    Returns
    -------
    genD : list
        List of genomic D information.
    
    """
    params_file = open(params_file_name, 'r')
    
    D_gene_info = {}

    in_D_gene_sec = False
    for line in params_file:
        if line.startswith('#GeneChoice;D_gene;'):
            in_D_gene_sec = True
        elif in_D_gene_sec:
            if line[0] == '%':
                split_line = line[1:].split(';')
                D_gene_info[split_line[0]] = [split_line[1] , int(split_line[2])]
            else:
                break
    params_file.close()
    
    genD = [[]]*len(D_gene_info.keys())
    
    for D_gene in D_gene_info.keys():
        genD[D_gene_info[D_gene][1]] = [D_gene, D_gene_info[D_gene][0]]

    return genD