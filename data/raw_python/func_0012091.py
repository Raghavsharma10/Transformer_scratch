def update_or_append_line(filename, prefix, new_line, keep_backup=True,
                          append=True):
    '''Search in file 'filename' for a line starting with 'prefix' and replace
    the line by 'new_line'.

    If a line starting with 'prefix' not exists 'new_line' will be appended.
    If the file not exists, it will be created.

    Return False if new_line was appended, else True (i.e. if the prefix was
    found within of the file).
    '''
    same_line_exists, line_updated = False, False
    filename = os.path.expanduser(filename)
    if os.path.isfile(filename):
        backup = filename + '~'
        shutil.move(filename, backup)
    #    with open(filename, 'w') as dest, open(backup, 'r') as source:
        with open(filename, 'w') as dest:
            with open(backup, 'r') as source:
                # try update..
                for line in source:
                    if line == new_line:
                        same_line_exists = True
                    if line.startswith(prefix):
                        dest.write(new_line + '\n')
                        line_updated = True
                    else:
                        dest.write(line)
                # ..or append
                if not (same_line_exists or line_updated) and append:
                    dest.write(new_line + '\n')
        if not keep_backup:
            os.remove(backup)
    else:
        with open(filename, 'w') as dest:
            dest.write(new_line + '\n')
    return same_line_exists or line_updated