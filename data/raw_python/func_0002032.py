def get_db_root():
    "Read dbroot folder from config and mkdir if required."
    d = get_config()
    dbroot = Path(d['pyciss_db']['path'])
    dbroot.mkdir(exist_ok=True)
    return dbroot