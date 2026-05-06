def fetch_and_convert_dataset(source_files, target_filename):
    """
    Decorator applied to a dataset conversion function that converts acquired
    source files into a dataset file that BatchUp can use.

    Parameters
    ----------
    source_file: list of `AbstractSourceFile` instances
        A list of files to be acquired
    target_filename: str or callable
        The name of the target file in which to store the converted data
        either as a string or as a function of the form `fn() -> str`
        that returns it.

    The conversion function is of the form `fn(source_paths, target_path)`.
    It should return `target_path` if successful, `None` otherwise.
    After the conversion function is successfully applied, the temporary
    source files that were downloaded or copied into BatchUp's temporary
    directory are deleted, unless the conversion function moved or deleted
    them in which case no action is taken.

    Example
    -------
    In this example, we will show how to acquire the USPS dataset from an
    online source. USPS is provided as an HDF5 file anyway, so the
    conversion function simply moves it to the target path:

    >>> import shutil
    >>>
    >>> _USPS_SRC_ONLINE = DownloadSourceFile(
    ...    filename='usps.h5',
    ...    url='https://github.com/Britefury/usps_dataset/raw/master/'
    ...        'usps.h5',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_ONLINE], 'usps.h5')
    ... def usps_data_online(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it:
    >>> usps_path = usps_data_online() #doctest: +SKIP

    In this example, the USPS dataset will be acquired from a file on the
    filesystem. Note that the source path is fixed; the next example
    shows how we can determine the source path dynamically:

    >>> _USPS_SRC_OFFLINE_FIXED = CopySourceFile(
    ...    filename='usps.h5',
    ...    source_path='some/path/to/usps.h5',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_OFFLINE_FIXED], 'usps.h5')
    ... def usps_data_offline_fixed(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it:
    >>> usps_path = usps_data_offline_fixed() #doctest: +SKIP

    The source path is provided as an argument to the decorated fetch
    function:

    >>> _USPS_SRC_OFFLINE_DYNAMIC = CopySourceFile(
    ...    filename='usps.h5',
    ...    arg_name='usps_path',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_OFFLINE_DYNAMIC], 'usps.h5')
    ... def usps_data_offline_dynamic(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it (note that the KW-arg `usps_path` is the same
    >>> # as the `arg_name` parameter given to `CopySourceFile` above:
    >>> usps_path = usps_data_offline_dynamic(
    ...    usps_path=get_config('mypath')) #doctest: +SKIP
    """
    if not isinstance(target_filename, six.string_types) and \
            not callable(target_filename):
        raise TypeError(
            'target_filename must either be a string or be callable (it is '
            'a {})'.format(type(target_filename)))

    for src in source_files:
        if not isinstance(src, AbstractSourceFile):
            raise TypeError('source_files should contain'
                            '`AbstractSourceFile` instances, '
                            'not {}'.format(type(src)))

    def decorate_fetcher(convert_function):
        def fetch(**kwargs):
            target_fn = path_string(target_filename)
            target_path = config.get_data_path(target_fn)

            # If the target file does not exist, we need to acquire the
            # source files and convert them
            if not os.path.exists(target_path):
                # Acquire the source files
                source_paths = []
                for src in source_files:
                    p = src.acquire(**kwargs)
                    if p is not None:
                        if p in source_paths:
                            raise ValueError(
                                'Duplicate source file {}'.format(p))
                        source_paths.append(p)
                    else:
                        print('Failed to acquire {}'.format(src))
                        return None

                # Got the source files
                # Convert
                converted_path = convert_function(source_paths, target_path)

                # If successful, delete the source files
                if converted_path is not None:
                    for src in source_files:
                        src.clean_up()

                return converted_path
            else:
                # Target file already exists
                return target_path

        fetch.__name__ = convert_function.__name__

        return fetch

    return decorate_fetcher