def remove_vcf_info(keyword, variant_line=None, variant_dict=None):
    """Remove the information of a info field of a vcf variant line or a 
        variant dict.
    
    Arguments:
        variant_line (str): A vcf formatted variant line
        variant_dict (dict): A variant dictionary
        keyword (str): The info field key
    
    Returns:
        variant_line (str): A annotated variant line
    """
    logger.debug("Removing variant information {0}".format(keyword))
    
    fixed_variant = None
    
    def get_new_info_string(info_string, keyword):
        """Return a info string without keyword info"""
        new_info_list = []
        splitted_info_string = info_string.split(';')
        for info in splitted_info_string:
            splitted_info_entry = info.split('=')
            if splitted_info_entry[0] != keyword:
                new_info_list.append(info)
        
        new_info_string = ';'.join(new_info_list)
        
        return new_info_string
        
    
    if variant_line:
        logger.debug("Removing information from a variant line")
        splitted_variant = variant_line.rstrip('\n').split('\t')
        
        old_info = splitted_variant[7]
        if old_info == '.':
            new_info_string = '.'
        else:
            new_info_string = get_new_info_string(old_info, keyword)
        
        splitted_variant[7] = new_info_string
        
        fixed_variant = '\t'.join(splitted_variant)
    
    elif variant_dict:
        logger.debug("Removing information to a variant dict")
        old_info = variant_dict['INFO']
        
        if old_info == '.':
            variant_dict['INFO'] = old_info
        else:
            new_info_string = get_new_info_string(old_info, keyword)
        
        variant_dict['INFO'] = new_info_string
        fixed_variant = variant_dict
    
    return fixed_variant