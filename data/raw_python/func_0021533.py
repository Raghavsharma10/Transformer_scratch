def read_igor_V_gene_parameters(params_file_name):
    """Load raw genV from file.
    
    genV is a list of genomic V information. Each element is a list of three 
    elements. The first is the name of the V allele, the second is the genomic 
    sequence trimmed to the CDR3 region for productive sequences, and the last 
    is the full germline sequence. For this 'raw genV' the middle element is an
    empty string to be filled in later.

    Parameters
    ----------
    params_file_name : str
        File name for a IGOR parameter file.

    Returns
    -------
    genV : list
        List of genomic V information.
    
    """
    params_file = open(params_file_name, 'r')
    
    V_gene_info = {}

    in_V_gene_sec = False
    for line in params_file:
        if line.startswith('#GeneChoice;V_gene;'):
            in_V_gene_sec = True
        elif in_V_gene_sec:
            if line[0] == '%':
                split_line = line[1:].split(';')
                V_gene_info[split_line[0]] = [split_line[1] , int(split_line[2])]
            else:
                break
    params_file.close()
    
    genV = [[]]*len(V_gene_info.keys())
    
    for V_gene in V_gene_info.keys():
        genV[V_gene_info[V_gene][1]] = [V_gene, '', V_gene_info[V_gene][0]]

    return genV