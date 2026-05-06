def update(connection=None, urls=None, force_download=False, taxids=None, silent=False):
    """Updates CTD database

    :param urls: list of urls to download
    :type urls: iterable
    :param connection: custom database connection string
    :type connection: str
    :param force_download: force method to download
    :type force_download: bool
    :param int,list,tuple taxids: int or iterable of NCBI taxonomy identifiers (default is None = load all)
    """
    if isinstance(taxids, int):
        taxids = (taxids,)
    db = DbManager(connection)
    db.db_import_xml(urls, force_download, taxids, silent)
    db.session.close()