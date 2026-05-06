def _travis_job_state(state):
    """ Converts a Travis state into a state character, color,
    and whether it's still running or a stopped state. """
    if state in [None, 'queued', 'created', 'received']:
        return colorama.Fore.YELLOW, '*', True
    elif state in ['started', 'running']:
        return colorama.Fore.LIGHTYELLOW_EX, '*', True
    elif state == 'passed':
        return colorama.Fore.LIGHTGREEN_EX, 'P', False
    elif state == 'failed':
        return colorama.Fore.LIGHTRED_EX, 'X', False
    elif state == 'errored':
        return colorama.Fore.LIGHTRED_EX, '!', False
    elif state == 'canceled':
        return colorama.Fore.LIGHTBLACK_EX, 'X', False
    else:
        raise RuntimeError('unknown state: %s' % str(state))