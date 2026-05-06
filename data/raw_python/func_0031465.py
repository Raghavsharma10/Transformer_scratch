def export_obo(path_to_file, connection=None):
    """export database to obo file

    :param path_to_file: path to export file
    :param connection: connection string (optional)
    :return:
    """
    db = DbManager(connection)
    db.export_obo(path_to_export_file=path_to_file)
    db.session.close()