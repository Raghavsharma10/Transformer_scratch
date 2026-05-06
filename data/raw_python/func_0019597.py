def write_equalwidth_bedfile(bedfile, width, outfile):
    """Read input from <bedfile>, set the width of all entries to <width> and 
    write the result to <outfile>.
    Input file needs to be in BED or WIG format."""

    BUFSIZE = 10000
    f = open(bedfile)
    out = open(outfile, "w")
    lines = f.readlines(BUFSIZE)
    line_count = 0
    while lines:
        for line in lines:
            line_count += 1
            if not line.startswith("#") and not line.startswith("track") and not line.startswith("browser"):
                vals = line.strip().split("\t")
                try:
                    start, end = int(vals[1]), int(vals[2])
                except ValueError:
                    print("Error on line %s while reading %s. Is the file in BED or WIG format?" % (line_count, bedfile))
                    sys.exit(1)

                start = (start + end) // 2 - (width // 2)
                # This shifts the center, but ensures the width is identical... maybe not ideal
                if start < 0:
                    start = 0
                end = start + width
                # Keep all the other information in the bedfile if it's there
                if len(vals) > 3:
                    out.write("%s\t%s\t%s\t%s\n" % (vals[0], start, end, "\t".join(vals[3:])))
                else:
                    out.write("%s\t%s\t%s\n" % (vals[0], start, end))
        lines = f.readlines(BUFSIZE)
    
    out.close()
    f.close()