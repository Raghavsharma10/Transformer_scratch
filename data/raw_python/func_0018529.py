def rename_backup(name, suffix='.bak'):
    """
    Append a backup prefix to a file or directory, with an increasing numeric
    suffix (.N) if a file already exists
    """
    newname = '%s%s' % (name, suffix)
    n = 0
    while os.path.exists(newname):
        n += 1
        newname = '%s%s.%d' % (name, suffix, n)
    log.info('Renaming %s to %s', name, newname)
    os.rename(name, newname)
    return newname