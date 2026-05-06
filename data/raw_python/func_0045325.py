def cli(ctx, vcf, verbose, outfile, silent):
    """Simple vcf operations"""
    # configure root logger to print to STDERR
    loglevel = LEVELS.get(min(verbose, 3))
    configure_stream(level=loglevel)
    
    if vcf == '-':
        handle = get_vcf_handle(fsock=sys.stdin)
    else:
        handle = get_vcf_handle(infile=vcf)
    
    head = HeaderParser()
    for line in handle:
        line = line.rstrip()
        if line.startswith('#'):
            if line.startswith('##'):
                head.parse_meta_data(line)
            else:
                head.parse_header_line(line)
        else:
            break
    ctx.head = head

    ctx.handle = itertools.chain([line], handle)
    ctx.outfile = outfile
    ctx.silent = silent