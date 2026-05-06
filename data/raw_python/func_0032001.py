def index_run(record_path, keep_json, check_duplicate):
    """
    Convert raw JSON records into sqlite3 DB.

    Normally RASH launches a daemon that takes care of indexing.
    See ``rash daemon --help``.

    """
    from .config import ConfigStore
    from .indexer import Indexer
    cfstore = ConfigStore()
    indexer = Indexer(cfstore, check_duplicate, keep_json, record_path)
    indexer.index_all()