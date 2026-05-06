def prepare_denovo_input_narrowpeak(inputfile, params, outdir):
    """Prepare a narrowPeak file for de novo motif prediction.

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

    bedfile = os.path.join(outdir, "input.from.narrowpeak.bed")
    p = re.compile(r'^(#|track|browser)')
    width = int(params["width"])
    logger.info("preparing input (narrowPeak to BED, width %s)", width)
    warn_no_summit = True
    with open(bedfile, "w") as f_out:
        with open(inputfile) as f_in:
            for line in f_in:
                if p.search(line):
                    continue
                vals = line.strip().split("\t")
                start, end = int(vals[1]), int(vals[2])
                summit = int(vals[9])
                if summit == -1:
                    if warn_no_summit:
                        logger.warn("No summit present in narrowPeak file, using the peak center.")
                        warn_no_summit = False
                    summit = (end - start) // 2

                start = start + summit - (width // 2)
                end = start + width
                f_out.write("{}\t{}\t{}\t{}\n".format(
                    vals[0],
                    start,
                    end,
                    vals[6]
                    ))
    
    prepare_denovo_input_bed(bedfile, params, outdir)