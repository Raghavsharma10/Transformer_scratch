def variants(ctx, snpeff):
    """Print the variants in a vcf"""
    head = ctx.parent.head
    vcf_handle = ctx.parent.handle
    outfile = ctx.parent.outfile
    silent = ctx.parent.silent
    
    print_headers(head, outfile=outfile, silent=silent)
    
    for line in vcf_handle:
        print_variant(variant_line=line, outfile=outfile, silent=silent)
        if snpeff:
            variant_dict =  get_variant_dict(
                variant_line = line,
                header_line = head.header
            )
            #Create a info dict:
            info_dict = get_info_dict(
                info_line = variant_dict['INFO']
            )
            snpeff_string = info_dict.get('ANN')

            if snpeff_string:
                #Get the snpeff annotations
                snpeff_info = get_snpeff_info(
                    snpeff_string = snpeff_string,
                    snpeff_header = head.snpeff_columns
                )