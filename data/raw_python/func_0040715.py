def print_results(results, outfile):
    """
    Write results to outfile
    """
    for stanza_words, scheme in results:
        outfile.write(str(' ').join(stanza_words) + str('\n'))
        outfile.write(str(' ').join(map(str, scheme)) + str('\n\n'))
    outfile.close()
    logging.info("Wrote result")