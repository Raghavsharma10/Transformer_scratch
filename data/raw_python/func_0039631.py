def echo_vacation_rc():
    """ Display all our .vacationrc file. """
    contents = rc.read()
    print('.vacationrc\n===========')
    for line in contents:
        print(line.rstrip())