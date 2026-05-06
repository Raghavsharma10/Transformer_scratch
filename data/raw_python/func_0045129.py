def get_vep_info(vep_string, vep_header):
    """Make the vep annotations into a dictionaries
    
        A vep dictionary will have the vep column names as keys and 
        the vep annotations as values.
        The dictionaries are stored in a list

        Args:
            vep_string (string): A string with the CSQ annotation
            vep_header (list): A list with the vep header
        
        Return:
            vep_annotations (list): A list of vep dicts
    
    """
    
    vep_annotations = [
        dict(zip(vep_header, vep_annotation.split('|'))) 
        for vep_annotation in vep_string.split(',')
    ]
    
    return vep_annotations