def _create_text_report(inputfile, motifs, closest_match, stats, outdir):
    """Create text report of motifs with statistics and database match."""
    my_stats = {}
    for motif in motifs:
        match = closest_match[motif.id]
        my_stats[str(motif)] = {}
        for bg in list(stats.values())[0].keys():
            if str(motif) not in stats:
                logger.error("####")
                logger.error("{} not found".format(str(motif)))
                for s in sorted(stats.keys()):
                    logger.error(s)
                logger.error("####")
            else:
                my_stats[str(motif)][bg] = stats[str(motif)][bg].copy()
                my_stats[str(motif)][bg]["best_match"] = "_".join(match[0].split("_")[:-1])
                my_stats[str(motif)][bg]["best_match_pvalue"] = match[1][-1]
    
    header = ("# GimmeMotifs version {}\n"
             "# Inputfile: {}\n"
             ).format(__version__, inputfile)

    write_stats(my_stats, os.path.join(outdir, "stats.{}.txt"), header=header)