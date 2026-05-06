def create_backgrounds(outdir, background=None, genome="hg38", width=200, custom_background=None):
    """Create different backgrounds for motif prediction and validation.

    Parameters
    ----------
    outdir : str
        Directory to save results.
    
    background : list, optional
        Background types to create, default is 'random'.

    genome : str, optional
        Genome name (for genomic and gc backgrounds).

    width : int, optional
        Size of background regions

    Returns
    -------
    bg_info : dict
        Keys: background name, values: file name.
    """
    if background is None:
        background = ["random"]
        nr_sequences = {}

    # Create background for motif prediction
    if "gc" in background:
        pred_bg = "gc"
    else:
        pred_bg = background[0]
    
    create_background(
                    pred_bg, 
                    os.path.join(outdir, "prediction.fa"), 
                    os.path.join(outdir, "prediction.bg.fa"), 
                    genome=genome, 
                    width=width,
                    custom_background=custom_background)

    # Get background fasta files for statistics
    bg_info = {}
    nr_sequences = {}    
    for bg in background:
        fname = os.path.join(outdir, "bg.{}.fa".format(bg))
        nr_sequences[bg] = create_background(
                                        bg, 
                                        os.path.join(outdir, "validation.fa"), 
                                        fname, 
                                        genome=genome, 
                                        width=width,
                                        custom_background=custom_background)

        bg_info[bg] = fname
    return bg_info