def search_run(output, **kwds):
    """
    Search command history.

    """
    from .config import ConfigStore
    from .database import DataBase
    from .query import expand_query, preprocess_kwds

    cfstore = ConfigStore()
    kwds = expand_query(cfstore.get_config(), kwds)
    format = get_formatter(**kwds)
    fmtkeys = formatter_keys(format)
    candidates = set([
        'command_count', 'success_count', 'success_ratio', 'program_count'])
    kwds['additional_columns'] = candidates & set(fmtkeys)

    db = DataBase(cfstore.db_path)
    for crec in db.search_command_record(**preprocess_kwds(kwds)):
        output.write(format.format(**crec.__dict__))