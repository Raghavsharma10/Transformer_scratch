def configure_logging(cfg, custom_level=None):
    """Don't know what this is for ...."""
    import itertools as it
    import operator as op

    if custom_level is None:
        custom_level = logging.WARNING
    for entity in it.chain.from_iterable(it.imap(op.methodcaller('viewvalues'),
                                                 [cfg] + [cfg.get(k, dict()) for k in ['handlers', 'loggers']])):
        if isinstance(entity, Mapping) and entity.get('level') == 'custom':
            entity['level'] = custom_level
    logging.config.dictConfig(cfg)
    logging.captureWarnings(cfg.warnings)