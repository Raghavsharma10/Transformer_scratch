def load_genomic_CDR3_anchor_pos_and_functionality(anchor_pos_file_name):
    """Read anchor position and functionality from file.

    Parameters
    ----------
    anchor_pos_file_name : str
        File name for the functionality and position of a conserved residue 
        that defines the CDR3 region for each V or J germline sequence.
        
    Returns
    -------
    anchor_pos_and_functionality : dict
        Residue anchor position and functionality for each gene/allele.
    
    """
    
    anchor_pos_and_functionality = {}
    anchor_pos_file = open(anchor_pos_file_name, 'r')
    
    first_line = True
    for line in anchor_pos_file:
        if first_line:
            first_line = False
            continue
        
        split_line = line.split(',')
        split_line = [x.strip() for x in split_line]
        anchor_pos_and_functionality[split_line[0]] = [int(split_line[1]), split_line[2].strip().strip('()')]

    return anchor_pos_and_functionality