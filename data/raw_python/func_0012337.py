def get_remaining_time(program):
    '''
    Get the remaining time in seconds of a program that is currently on.
    '''
    now = datetime.datetime.now()
    program_start = program.get('start_time')
    program_end = program.get('end_time')
    if not program_start or not program_end:
        _LOGGER.error('Could not determine program start and/or end times.')
        _LOGGER.debug('Program data: %s', program)
        return
    if now > program_end:
        _LOGGER.error('The provided program has already ended.')
        _LOGGER.debug('Program data: %s', program)
        return 0
    progress = now - program_start
    return progress.seconds