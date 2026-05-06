def int_input(message, low, high, show_range = True):
    '''
    Ask a user for a int input between two values

    args:
        message (str): Prompt for user
        low (int): Low value, user entered value must be > this value to be accepted
        high (int): High value, user entered value must be < this value to be accepted
        show_range (boolean, Default True): Print hint to user the range

    returns:
        int_in (int): Input integer
    '''

    int_in = low - 1
    while (int_in < low) or (int_in > high):
        if show_range:
            suffix = ' (integer between ' + str(low) + ' and ' + str(high) + ')'
        else:
            suffix = ''
        inp = input('Enter a ' + message + suffix + ': ')
        if re.match('^-?[0-9]+$', inp) is not None:
            int_in = int(inp)
        else:
            print(colored('Must be an integer, try again!', 'red'))
    return int_in