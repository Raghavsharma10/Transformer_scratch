def read_igor_J_gene_parameters(params_file_name):
    """Load raw genJ from file.
    
    genJ is a list of genomic J information. Each element is a list of three 
    elements. The first is the name of the J allele, the second is the genomic 
    sequence trimmed to the CDR3 region for productive sequences, and the last 
    is the full germline sequence. For this 'raw genJ' the middle element is an
    empty string to be filled in later.

    Parameters
    ----------
    params_file_name : str
        File name for a IGOR parameter file.

    Returns
    -------
    genJ : list
        List of genomic J information.
    
    """
    params_file = open(params_file_name, 'r')
    
    J_gene_info = {}

    in_J_gene_sec = False
    for line in params_file:
        if line.startswith('#GeneChoice;J_gene;'):
            in_J_gene_sec = True
        elif in_J_gene_sec:
            if line[0] == '%':
                split_line = line[1:].split(';')
                J_gene_info[split_line[0]] = [split_line[1] , int(split_line[2])]
            else:
                break
    params_file.close()
    
    genJ = [[]]*len(J_gene_info.keys())
    
    for J_gene in J_gene_info.keys():
        genJ[J_gene_info[J_gene][1]] = [J_gene, '', J_gene_info[J_gene][0]]

    return genJ