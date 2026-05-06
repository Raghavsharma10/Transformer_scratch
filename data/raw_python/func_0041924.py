def append(entry):
    """ Append either a list of strings or a string to our file. """
    if not entry:
        return
    try:
        with open(get_rc_path(), 'a') as f:
            if isinstance(entry, list):
                f.writelines(entry)
            else:
                f.write(entry + '\n')
    except IOError:
        print('Error writing your ~/.vacationrc file!')