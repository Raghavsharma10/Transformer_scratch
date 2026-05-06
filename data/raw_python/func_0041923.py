def write(entries):
    """ Write an entire rc file. """
    try:
        with open(get_rc_path(), 'w') as rc:
            rc.writelines(entries)
    except IOError:
        print('Error writing your ~/.vacationrc file!')