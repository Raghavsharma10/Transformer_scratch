def add_vcf_info(keyword, variant_line=None, variant_dict=None, annotation=None):
    """
    Add information to the info field of a vcf variant line.
    
    Arguments:
        variant_line (str): A vcf formatted variant line
        keyword (str): The info field key
        annotation (str): If the annotation is a key, value pair
                          this is the string that represents the value
    
    Returns:
        fixed_variant : str if variant line, or dict if variant_dict
    """
    logger = logging.getLogger(__name__)
    
    if annotation:
        new_info = '{0}={1}'.format(keyword, annotation)
    else:
        new_info = keyword
    
    logger.debug("Adding new variant information {0}".format(new_info))
    
    fixed_variant = None
    
    if variant_line:
        logger.debug("Adding information to a variant line")
        splitted_variant = variant_line.rstrip('\n').split('\t')
        logger.debug("Adding information to splitted variant line")
        old_info = splitted_variant[7]
        if old_info == '.':
            splitted_variant[7] = new_info
        else:
            splitted_variant[7] = "{0};{1}".format(splitted_variant[7], new_info)
        
        fixed_variant = '\t'.join(splitted_variant)
    
    elif variant_dict:
        logger.debug("Adding information to a variant dict")
        old_info = variant_dict['INFO']
        if old_info == '.':
            variant_dict['INFO'] = new_info
        else:
            variant_dict['INFO'] = "{0};{1}".format(old_info, new_info)
        fixed_variant = variant_dict
    
    return fixed_variant