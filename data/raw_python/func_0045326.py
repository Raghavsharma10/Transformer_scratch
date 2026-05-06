def delete_info(ctx, info):
    """Delete a info field from all variants in a vcf"""
    head = ctx.parent.head
    vcf_handle = ctx.parent.handle
    outfile = ctx.parent.outfile
    silent = ctx.parent.silent
    
    if not info:
        logger.error("No info provided")
        sys.exit("Please provide a info string")
    
    if not info in head.info_dict:
        logger.error("Info '{0}' is not specified in vcf header".format(info))
        sys.exit("Please provide a valid info field")
    
    head.remove_header(info)
    
    print_headers(head, outfile=outfile, silent=silent)
    
    for line in vcf_handle:
        line = line.rstrip()
        new_line = remove_vcf_info(keyword=info, variant_line=line)
        print_variant(variant_line=new_line, outfile=outfile, silent=silent)