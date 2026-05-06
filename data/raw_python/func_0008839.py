def load_sources(filename):
    """
    Open a file, read contents, return a list of all the sources in that file.
    @param filename:
    @return: list of OutputSource objects
    """
    catalog = catalogs.table_to_source_list(catalogs.load_table(filename))
    logging.info("read {0} sources from {1}".format(len(catalog), filename))
    return catalog