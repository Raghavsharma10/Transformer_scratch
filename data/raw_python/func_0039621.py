def execute(tokens):
    """ Perform the actions described by the input tokens. """
    if not validate_rc():
        print('Your .vacationrc file has errors!')
        echo_vacation_rc()
        return

    for action, value in tokens:
        if action == 'show':
            show()
        elif action == 'log':
            log_vacation_days()
        elif action == 'echo':
            echo_vacation_rc()
        elif action == 'take':
            take(value)
        elif action == 'cancel':
            cancel(value)
        elif action == 'setrate':
            setrate(value)
        elif action == 'setdays':
            setdays(value)