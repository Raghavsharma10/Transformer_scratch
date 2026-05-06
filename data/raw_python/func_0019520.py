def check_denovo_input(inputfile, params):
    """
    Check if an input file is valid, which means BED, narrowPeak or FASTA
    """
    background = params["background"]
    
    input_type = determine_file_type(inputfile)
    
    if input_type == "fasta":
        valid_bg = FA_VALID_BGS    
    elif input_type in ["bed", "narrowpeak"]:
        genome = params["genome"]
        valid_bg = BED_VALID_BGS    
        if "genomic" in background or "gc" in background:
            Genome(genome)
        # is it a valid bed-file etc.
        check_bed_file(inputfile)    # bed-specific, will also work for narrowPeak
    else:
        sys.stderr.write("Format of inputfile {} not recognized.\n".format(inputfile))
        sys.stderr.write("Input should be FASTA, BED or narrowPeak.\n")
        sys.stderr.write("See https://genome.ucsc.edu/FAQ/FAQformat.html for specifications.\n")
        sys.exit(1)

    for bg in background:
        if not bg in valid_bg:
            logger.info("Input type is %s, ignoring background type '%s'", 
                            input_type, bg)
        background = [bg for bg in background if bg in valid_bg]

    if len(background) == 0:
        logger.error("No valid backgrounds specified!")
        sys.exit(1)

    return input_type, background