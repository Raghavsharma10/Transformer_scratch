def read_file(path, encoding='utf-8', *args, **kwargs):
    ''' Read text file content. If the file name ends with .gz, read it as gzip file.
    If mode argument is provided as 'rb', content will be read as byte stream.
    By default, content is read as string.
    '''
    if 'mode' in kwargs and kwargs['mode'] == 'rb':
        return process_file(path, processor=lambda x: x.read(),
                            encoding=encoding, *args, **kwargs)
    else:
        return process_file(path, processor=lambda x: to_string(x.read(), encoding),
                            encoding=encoding, *args, **kwargs)