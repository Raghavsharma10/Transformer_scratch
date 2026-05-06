def uri_open(uri, mode='rb', auto_compress=True, in_memory=True, delete_tempfile=True, textio_args={}, storage_args={}):
    """
    Opens a URI for reading / writing.
    Analogous to the :func:`open` function.
    This method supports ``with`` context handling::

        with uri_open('http://www.example.com', mode='r') as f:
            print(f.read())

    :param str uri: URI of file to open
    :param str mode: Either ``rb``, ``r``, ``w``, or ``wb`` for read/write modes in binary/text respectiely
    :param bool auto_compress: Whether to automatically use the :mod:`gzip` module with ``.gz`` URIsF
    :param bool in_memory: Whether to store entire file in memory or in a local temporary file
    :param bool delete_tempfile: When :attr:`in_memory` is ``False``, whether to delete the temporary file on close
    :param dict textio_args: Keyword arguments to pass to :class:`io.TextIOWrapper` for text read/write mode
    :param dict storage_args: Keyword arguments to pass to the underlying storage object

    :returns: file-like object to URI
    """

    if isinstance(uri, BaseURI): uri = str(uri)
    uri_obj = get_uri_obj(uri, storage_args)

    if mode == 'rb': read_mode, binary_mode = True, True
    elif mode == 'r': read_mode, binary_mode = True, False
    elif mode == 'w': read_mode, binary_mode = False, False
    elif mode == 'wb': read_mode, binary_mode = False, True
    else: raise TypeError('`mode` cannot be "{}".'.format(mode))

    if read_mode:
        if in_memory:
            file_obj = BytesIO(uri_obj.get_content())
            setattr(file_obj, 'name', str(uri_obj))
        else:
            file_obj = _TemporaryURIFileIO(uri_obj=uri_obj, input_mode=True, delete_tempfile=delete_tempfile)
        #end if
    else:
        if in_memory: file_obj = URIBytesOutput(uri_obj)
        else:
            file_obj = _TemporaryURIFileIO(uri_obj=uri_obj, input_mode=False, pre_close_action=uri_obj.upload_file, delete_tempfile=delete_tempfile)
            setattr(file_obj, 'name', str(uri_obj))
        #end if
    #end if

    temp_name = getattr(file_obj, 'temp_name', None)

    if auto_compress:
        _, ext = os.path.splitext(uri)
        ext = ext.lower()
        if ext == '.gz': file_obj = gzip.GzipFile(fileobj=file_obj, mode='rb' if read_mode else 'wb')
    #end if

    if not binary_mode:
        textio_args.setdefault('encoding', 'utf-8')
        file_obj = TextIOWrapper(file_obj, **textio_args)
    #end if

    if not hasattr(file_obj, 'temp_name'): setattr(file_obj, 'temp_name', temp_name)

    return file_obj