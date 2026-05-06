def validate_rc():
    """ Before we execute any actions, let's validate our .vacationrc. """
    transactions = rc.read()
    if not transactions:
        print('Your .vacationrc file is empty! Set days and rate.')
        return False
    transactions = sort(unique(transactions))
    return validate_setup(transactions)