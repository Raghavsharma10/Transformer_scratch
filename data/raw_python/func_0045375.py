def get_chromosome_priority(chrom, chrom_dict={}):
    """
    Return the chromosome priority
    
    Arguments:
        chrom (str): The cromosome name from the vcf
        chrom_dict (dict): A map of chromosome names and theis priority
    
    Return:
        priority (str): The priority for this chromosom
    """
    priority = 0
    
    chrom = str(chrom).lstrip('chr')
    
    if chrom_dict:
        priority = chrom_dict.get(chrom, 0)
    
    else:
        try:
            if int(chrom) < 23:
                priority = int(chrom)
        except ValueError:
            if chrom == 'X':
                priority = 23
            elif chrom == 'Y':
                priority = 24
            elif chrom == 'MT':
                priority = 25
            else:
                priority = 26
    
    return str(priority)