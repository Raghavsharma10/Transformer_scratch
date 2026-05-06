def get_label_map(opts):
    ''' Find volume labels from filesystem and return in dict format. '''
    result = {}
    try:  # get labels from filesystem
        for entry in os.scandir(diskdir):
            if entry.name.startswith('.'):
                continue
            if islink(entry.path):
                target = os.readlink(entry.path)
            else:
                target = entry.path
            result[target] = entry.name
        if opts.debug:
            print('\n\nlabel_map:', result)
    except FileNotFoundError:
        pass

    return result