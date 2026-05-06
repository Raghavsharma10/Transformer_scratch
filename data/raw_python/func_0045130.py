def get_snpeff_info(snpeff_string, snpeff_header):
    """Make the vep annotations into a dictionaries
    
        A snpeff dictionary will have the snpeff column names as keys and 
        the vep annotations as values.
        The dictionaries are stored in a list. 
        One dictionary for each transcript.

        Args:
            snpeff_string (string): A string with the ANN annotation
            snpeff_header (list): A list with the vep header
        
        Return:
            snpeff_annotations (list): A list of vep dicts
    
    """
    
    snpeff_annotations = [
        dict(zip(snpeff_header, snpeff_annotation.split('|'))) 
        for snpeff_annotation in snpeff_string.split(',')
    ]
    
    return snpeff_annotations