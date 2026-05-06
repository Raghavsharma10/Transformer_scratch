def touch():
    """ Create a .vacationrc file if none exists. """
    if not os.path.isfile(get_rc_path()):
        open(get_rc_path(), 'a').close()
        print('Created file: {}'.format(get_rc_path()))