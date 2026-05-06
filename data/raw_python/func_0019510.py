def write_stats(stats, fname, header=None):
    """write motif statistics to text file."""
    # Write stats output to file

    for bg in list(stats.values())[0].keys():
        f = open(fname.format(bg), "w")
        if header:
            f.write(header)
        
        stat_keys = sorted(list(list(stats.values())[0].values())[0].keys())
        f.write("{}\t{}\n".format("Motif", "\t".join(stat_keys)))

        for motif in stats:
            m_stats = stats.get(str(motif), {}).get(bg)
            if m_stats:
                f.write("{}\t{}\n".format(
                    "_".join(motif.split("_")[:-1]),
                    "\t".join([str(m_stats[k]) for k in stat_keys])
                    ))
            else:
                logger.warn("No stats for motif {0}, skipping this motif!".format(motif.id))
            #motifs.remove(motif)
        f.close()

    return