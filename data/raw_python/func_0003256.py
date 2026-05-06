def syscall_direct(*events):
    '''
    Directly process these events. This should never be used for normal events.
    '''
    def _syscall(scheduler, processor):
        for e in events:
            processor(e)
    return _syscall