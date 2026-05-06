def get_label_map(opts):
    ''' Find volume labels from filesystem and return in dict format. '''
    results = {}
    try:
        for entry in os.scandir(diskdir):
            target = normpath(join(diskdir, os.readlink(entry.path)))
            decoded_name = entry.name.encode('utf8').decode('unicode_escape')
            results[target] = decoded_name
        if opts.debug:
            print('\n\nlabel_map:', results)
    except FileNotFoundError:
        pass
    return results