def bool_input(message):
    '''
    Ask a user for a boolean input

    args:
        message (str): Prompt for user

    returns:
        bool_in (boolean): Input boolean
    '''

    while True:
        suffix = ' (true or false): '
        inp = input(message + suffix)
        if inp.lower() == 'true':
            return True
        elif inp.lower() == 'false':
            return False
        else:
            print(colored('Must be either true or false, try again!', 'red'))