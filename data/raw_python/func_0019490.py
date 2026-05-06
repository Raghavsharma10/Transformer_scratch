def prepare_denovo_input_bed(inputfile, params, outdir):
    """Prepare a BED file for de novo motif prediction.

    All regions to same size; split in test and validation set;
    converted to FASTA.

    Parameters
    ----------
    inputfile : str
        BED file with input regions.

    params : dict
        Dictionary with parameters.

    outdir : str
        Output directory to save files.
    """
    logger.info("preparing input (BED)")
    
    # Create BED file with regions of equal size
    width = int(params["width"])
    bedfile = os.path.join(outdir, "input.bed")
    write_equalwidth_bedfile(inputfile, width, bedfile)
    
    abs_max = int(params["abs_max"])
    fraction = float(params["fraction"])
    pred_bedfile = os.path.join(outdir, "prediction.bed")
    val_bedfile = os.path.join(outdir, "validation.bed")
    # Split input into prediction and validation set
    logger.debug(
                "Splitting %s into prediction set (%s) and validation set (%s)",
                bedfile, pred_bedfile, val_bedfile)
    divide_file(bedfile, pred_bedfile, val_bedfile, fraction, abs_max)

    config = MotifConfig()
   
    genome = Genome(params["genome"])
    for infile in [pred_bedfile, val_bedfile]:
        genome.track2fasta(
            infile, 
            infile.replace(".bed", ".fa"), 
            )

    # Create file for location plots
    lwidth = int(params["lwidth"])
    extend = (lwidth - width) // 2
    
    genome.track2fasta(
            val_bedfile, 
            os.path.join(outdir, "localization.fa"), 
            extend_up=extend, 
            extend_down=extend, 
            stranded=params["use_strand"], 
            )