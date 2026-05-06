def get_current_program_progress(program):
    '''
    Get the current progress of the program in %
    '''
    now = datetime.datetime.now()
    program_duration = get_program_duration(program)
    if not program_duration:
        return
    progress = now - program.get('start_time')
    return progress.seconds * 100 / program_duration