def write_file(path, content, mode=None, encoding='utf-8'):
    ''' Write content to a file. If the path ends with .gz, gzip will be used. '''
    if not mode:
        if isinstance(content, bytes):
            mode = 'wb'
        else:
            mode = 'wt'
    if not path:
        raise ValueError("Output path is invalid")
    else:
        getLogger().debug("Writing content to {}".format(path))
        # convert content to string when writing text data
        if mode in ('w', 'wt') and not isinstance(content, str):
            content = to_string(content)
        elif mode == 'wb':
            # content needs to be encoded as bytes
            if not isinstance(content, str):
                content = to_string(content).encode(encoding)
            else:
                content = content.encode(encoding)
        if str(path).endswith('.gz'):
            with gzip.open(path, mode) as outfile:
                outfile.write(content)
        else:
            with open(path, mode=mode) as outfile:
                outfile.write(content)