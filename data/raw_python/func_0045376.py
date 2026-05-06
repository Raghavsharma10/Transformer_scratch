def sort_variants(vcf_handle):
    """Sort the variants of a vcf file
    
        Args:
            vcf_handle
            mode (str): position or rank score
        
        Returns:
            sorted_variants (Iterable): An iterable with sorted variants
    """
    logger.debug("Creating temp file")
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.close()
    logger.debug("Opening temp file with codecs")
    temp_file_handle = codecs.open(
                        temp_file.name,
                        mode='w',
                        encoding='utf-8',
                        errors='replace'
                        )

    try:
        with codecs.open(temp_file.name,mode='w',encoding='utf-8',errors='replace') as f:
            for line in vcf_handle:
                if not line.startswith('#'):
                    line = line.rstrip().split('\t')
                    chrom = line[0]
                    priority = get_chromosome_priority(chrom)
                
                    print_line = "{0}\t{1}\n".format(priority, '\t'.join(line))
                    f.write(print_line)
        #Sort the variants
        sort_variant_file(temp_file.name)
        
        with codecs.open(temp_file.name,mode='r',encoding='utf-8',errors='replace') as f:
            for line in f:
                line = line.rstrip().split('\t')
                yield '\t'.join(line[1:])

    except Exception as err:
        logger.error("Something went wrong")
        logger.error(err)
    finally:
        logger.debug("Deleting temp file")
        os.remove(temp_file.name)
        logger.debug("Temp file deleted")