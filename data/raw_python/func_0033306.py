def session(connection_string=None):
    """Gets a SQLAlchemy session"""
    global _session_makers
    connection_string = connection_string or oz.settings["db"]

    if not connection_string in _session_makers:
        _session_makers[connection_string] = sessionmaker(bind=engine(connection_string=connection_string))

    return _session_makers[connection_string]()