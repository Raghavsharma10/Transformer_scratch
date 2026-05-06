def update(connection=None, urls=None, force_download=False):
    """Updates CTD database

    :param iter[str] urls: list of urls to download
    :param str connection: custom database connection string
    :param bool force_download: force method to download
    """
    db = DbManager(connection)
    db.db_import(urls=urls, force_download=force_download)
    db.session.close()