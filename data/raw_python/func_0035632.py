def load_db_from_url(url="https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"):
    """ Loads the database from a gzipped version of the system folder, by default the one located in the oec_gzip repo
    in the OpenExoplanetCatalogue GitHub group.

    The database is loaded from the url in memory

    :param url: url to load (must be gzipped version of systems folder)
    :return: OECDatabase objected initialised with latest OEC Version
    """

    catalogue = gzip.GzipFile(fileobj=io.BytesIO(requests.get(url).content))
    database = OECDatabase(catalogue, stream=True)

    return database