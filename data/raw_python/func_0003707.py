def float_input(message, low, high):
    '''
    Ask a user for a float input between two values

    args:
        message (str): Prompt for user
        low (float): Low value, user entered value must be > this value to be accepted
        high (float): High value, user entered value must be < this value to be accepted

    returns:
        float_in (int): Input float
    '''

    float_in = low - 1.0
    while (float_in < low) or (float_in > high):
        inp = input('Enter a ' + message + ' (float between ' + str(low) + ' and ' + str(high) + '): ')
        if re.match('^([0-9]*[.])?[0-9]+$', inp) is not None:
            float_in = float(inp)
        else:
            print(colored('Must be a float, try again!', 'red'))
    return float_in