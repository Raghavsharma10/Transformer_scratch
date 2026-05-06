async def async_get_current_program(channel, no_cache=False):
    '''
    Get the current program info
    '''
    chan = await async_determine_channel(channel)
    guide = await async_get_program_guide(chan, no_cache)
    if not guide:
        _LOGGER.warning('Could not retrieve TV program for %s', channel)
        return
    now = datetime.datetime.now()
    for prog in guide:
        start = prog.get('start_time')
        end = prog.get('end_time')
        if now > start and now < end:
            return prog