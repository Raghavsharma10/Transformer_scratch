def get_variant_dict(variant_line, header_line=None):
    """Parse a variant line
        
        Split a variant line and map the fields on the header columns
        
        Args:
            variant_line (str): A vcf variant line
            header_line (list): A list with the header columns
        Returns:
            variant_dict (dict): A variant dictionary
    """
    if not header_line:
        logger.debug("No header line, use only first 8 mandatory fields")
        header_line = ['CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO']
    
    logger.debug("Building variant dict from variant line {0} and header"\
    " line {1}".format(variant_line, '\t'.join(header_line)))
    
    splitted_line = variant_line.rstrip().split('\t')
    if len(splitted_line) < len(header_line):
        logger.info('\t'.join(header_line))
        logger.info('\t'.join(splitted_line))
        raise SyntaxError("Length of variant line differs from length of"\
                            " header line")
    
    return dict(zip(header_line, splitted_line))