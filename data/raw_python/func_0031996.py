def update_data(func):
    """
    Decorator to save data more easily. Use parquet as data format

    Args:
        func: function to load data from data source

    Returns:
        wrapped function
    """
    default = dict([
        (param.name, param.default)
        for param in inspect.signature(func).parameters.values()
        if param.default != getattr(inspect, '_empty')
    ])

    @wraps(func)
    def wrapper(*args, **kwargs):

        default.update(kwargs)
        kwargs.update(default)
        cur_mod = sys.modules[func.__module__]
        logger = logs.get_logger(name_or_func=f'{cur_mod.__name__}.{func.__name__}', types='stream')

        root_path = cur_mod.DATA_PATH
        date_type = kwargs.pop('date_type', 'date')
        save_static = kwargs.pop('save_static', True)
        save_dynamic = kwargs.pop('save_dynamic', True)
        symbol = kwargs.get('symbol')
        file_kw = dict(func=func, symbol=symbol, root=root_path, date_type=date_type)
        d_file = cache_file(has_date=True, **file_kw)
        s_file = cache_file(has_date=False, **file_kw)

        cached = kwargs.pop('cached', False)
        if cached and save_static and files.exists(s_file):
            logger.info(f'Reading data from {s_file} ...')
            return pd.read_parquet(s_file)

        data = func(*args, **kwargs)

        if save_static:
            files.create_folder(s_file, is_file=True)
            save_data(data=data, file_fmt=s_file, append=False)
            logger.info(f'Saved data file to {s_file} ...')

        if save_dynamic:
            drop_dups = kwargs.pop('drop_dups', None)
            files.create_folder(d_file, is_file=True)
            save_data(data=data, file_fmt=d_file, append=True, drop_dups=drop_dups)
            logger.info(f'Saved data file to {d_file} ...')

        return data

    return wrapper