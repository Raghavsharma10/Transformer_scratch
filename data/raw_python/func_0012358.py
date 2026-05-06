def save_load_result(func):
    """Saves and/or loads func output (must be picklable)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Default behavior is no saving and loading. Specify save_name to save
        and load.

        Parameters
        ----------
        save_name: str, optional
            File name including directory and excluding extension.
        save: bool, optional
            Whether or not to save.
        load: bool, optional
            Whether or not to load.
        overwrite existing: bool, optional
            When the save path already contains file: if True, file will be
            overwritten, if False the data will be saved with the system time
            appended to the file name.
        warn_if_error: bool, optional
            Whether or not to issue UserWarning if load=True and save_name
            is not None but there is an error loading.

        Returns
        -------
        Result
            func output.
        """
        save_name = kwargs.pop('save_name', None)
        save = kwargs.pop('save', save_name is not None)
        load = kwargs.pop('load', save_name is not None)
        overwrite_existing = kwargs.pop('overwrite_existing', True)
        warn_if_error = kwargs.pop('warn_if_error', False)
        if load:
            if save_name is None:
                warnings.warn(
                    ('{} has load=True but cannot load because '
                     'save_name=None'.format(func.__name__)),
                    UserWarning)
            else:
                try:
                    return pickle_load(save_name)
                except (OSError, IOError) as err:
                    if warn_if_error:
                        msg = ('{} had {} loading file {}.'.format(
                            func.__name__, type(err).__name__, save_name))
                        msg = ' Continuing without loading.'
                        warnings.warn(msg, UserWarning)
        result = func(*args, **kwargs)
        if save:
            if save_name is None:
                warnings.warn((func.__name__ + ' has save=True but cannot ' +
                               'save because save_name=None'), UserWarning)
            else:
                pickle_save(result, save_name,
                            overwrite_existing=overwrite_existing)
        return result
    return wrapper