def get_program_duration(program):
    '''
    Get a program's duration in seconds
    '''
    program_start = program.get('start_time')
    program_end = program.get('end_time')
    if not program_start or not program_end:
        _LOGGER.error('Could not determine program start and/or end times.')
        _LOGGER.debug('Program data: %s', program)
        return
    program_duration = program_end - program_start
    return program_duration.seconds