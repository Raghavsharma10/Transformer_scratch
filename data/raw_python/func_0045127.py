def get_info_dict(info_line):
    """Parse a info field of a variant
        
        Make a dictionary from the info field of a vcf variant.
        Keys are the info keys and values are the raw strings from the vcf
        If the field only have a key (no value), value of infodict is True.
        
        Args:
            info_line (str): The info field of a vcf variant
        Returns:
            info_dict (dict): A INFO dictionary
    """
    
    variant_info = {}
    for raw_info in info_line.split(';'):
        splitted_info = raw_info.split('=')
        if len(splitted_info) == 2:
            variant_info[splitted_info[0]] = splitted_info[1]
        else:
            variant_info[splitted_info[0]] = True
    
    return variant_info