def syscall_generator(generator):
    '''
    Directly process events from a generator function. This should never be used for normal events.
    '''
    def _syscall(scheduler, processor):
        for e in generator():
            processor(e)
    return _syscall