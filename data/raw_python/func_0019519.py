def check_bed_file(fname):
    """ Check if the inputfile is a valid bed-file """
    if not os.path.exists(fname):
        logger.error("Inputfile %s does not exist!", fname)
        sys.exit(1)

    for i, line in enumerate(open(fname)):
        if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
            # comment or BED specific stuff
            pass
        else:
            vals = line.strip().split("\t")
            if len(vals) < 3:
                logger.error("Expecting tab-seperated values (chromosome<tab>start<tab>end) on line %s of file %s", i + 1, fname)
                sys.exit(1)
            try:
                start, end = int(vals[1]), int(vals[2])
            except ValueError:
                logger.error("No valid integer coordinates on line %s of file %s", i + 1, fname)
                sys.exit(1)
            if len(vals) > 3:
                try:
                    float(vals[3])
                except ValueError:
                    pass