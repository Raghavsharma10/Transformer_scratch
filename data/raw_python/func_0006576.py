def init_module(remote_credences=None,local_path=None):
    """Connnexion informations : remote_credences for remote acces OR local_path for local access"""
    if remote_credences is not None:
        RemoteConnexion.HOST = remote_credences["DB"]["host"]
        RemoteConnexion.USER = remote_credences["DB"]["user"]
        RemoteConnexion.PASSWORD = remote_credences["DB"]["password"]
        RemoteConnexion.NAME = remote_credences["DB"]["name"]
        MonoExecutant.ConnectionClass = RemoteConnexion
        Executant.ConnectionClass = RemoteConnexion
        abstractRequetesSQL.setup_marks("psycopg2")
    elif local_path is not None:
        LocalConnexion.PATH = local_path
        MonoExecutant.ConnectionClass = LocalConnexion
        Executant.ConnectionClass = LocalConnexion
        abstractRequetesSQL.setup_marks("sqlite3")
    else:
        raise ValueError("Sql module should be init with one of remote or local mode !")
    logging.info(f"Sql module initialized with {MonoExecutant.ConnectionClass.__name__}")