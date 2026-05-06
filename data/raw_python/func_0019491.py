def prepare_denovo_input_fa(inputfile, params, outdir):
    """Create all the FASTA files for de novo motif prediction and validation.
    
    Parameters
    ----------
    """
    fraction = float(params["fraction"])
    abs_max = int(params["abs_max"])

    logger.info("preparing input (FASTA)")

    pred_fa = os.path.join(outdir, "prediction.fa")
    val_fa = os.path.join(outdir, "validation.fa")
    loc_fa = os.path.join(outdir, "localization.fa")

    # Split inputfile in prediction and validation set
    logger.debug(
        "Splitting %s into prediction set (%s) and validation set (%s)",
        inputfile, pred_fa, val_fa)

    divide_fa_file(inputfile, pred_fa, val_fa, fraction, abs_max)

    # File for location plots
    shutil.copy(val_fa, loc_fa)
    seqs = Fasta(loc_fa).seqs
    lwidth = len(seqs[0])
    all_same_width = not(False in [len(seq) == lwidth for seq in seqs])
    if not all_same_width:
        logger.warn(
            "PLEASE NOTE: FASTA file contains sequences of different lengths. "
            "Positional preference plots might be incorrect!")